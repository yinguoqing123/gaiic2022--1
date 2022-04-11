import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoConfig, AutoTokenizer
from torch.utils.data import DataLoader

class MMoE(nn.Module):
    def __init__(self, num_experts=10, num_tasks=12, dims=768) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_spe = 1
        self.num_tasks = num_tasks
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(dims, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU()) 
                                      for i in range(num_experts)])
        self.experts_spe = nn.ModuleDict({f'task_{task_index}': nn.ModuleList([nn.Sequential(nn.Linear(dims, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU()) for _ in range(self.num_experts_spe)]) for task_index in range(self.num_tasks)})
        # self.gates = nn.ModuleList([nn.Sequential(nn.Linear(dims, 128), nn.ReLU(), nn.Linear(128, self.num_experts+self.num_experts_spe)) for i in range(num_tasks)])
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(dims, self.num_experts+self.num_experts_spe, bias=False)) for i in range(num_tasks)])
        self.label_nums = [14, 4, 3, 2, 3, 2, 4, 7, 4, 3, 6, 2]
        self.classifies = nn.ModuleList([nn.Linear(128, num) for num in self.label_nums])
        
    def forward(self, input):
        out_experts = []
        for layer in self.experts:
            out_experts.append(layer(input))  # num_experts, b, 128

        out_experts = torch.stack(out_experts, dim=0).permute(1, 0, 2) # b, num_experts, 128
        
        out_experts_spe = []
        for task_index in range(self.num_tasks):
            for layer in self.experts_spe[f'task_{task_index}']:
                out_experts_spe.append(layer(input))   # num_task * num_experts_spe, bsz, dim
                
        out_experts_spe = torch.stack(out_experts_spe, dim=0).reshape(self.num_tasks, self.num_experts_spe, -1, 128).permute(2, 0, 1, 3)  #  bsz, num_task, num_experts_spe, dim
        bsz, num_task, num_experts_spe, dim = out_experts_spe.size()

        out_experts = torch.cat([out_experts.unsqueeze(dim=1).expand(-1, num_task, -1, -1), out_experts_spe], dim=2)
        
        out_gates = []
        for layer in self.gates:
            out_gates.append(layer(input))  # num_tasks, b, num_experts
            
        out_gates = torch.stack(out_gates, dim=0).permute(1, 0, 2)  # b, num_tasks, num_experts+num_experts_spe
        out_gates = torch.softmax(out_gates, dim=-1)

        out = out_experts * out_gates.unsqueeze(dim=-1)  # b, num_tasks,  num_experts+num_experts_spe, 128
        out = torch.sum(out, dim=2)  # b, num_tasks, 128
        
        final = []
        for i in range(len(self.label_nums)):
            prob = self.classifies[i](out[:, i, :])
            prob = torch.softmax(prob, dim=-1)
            final.append(prob)
            
        return final  #  num_tasks, batch_size , (task_category_num)  (列表)

    
class ClassifyLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.losses = nn.ModuleList([nn.CrossEntropyLoss() for i in range(12)])
        
    def forward(self, input, target, mask):
        mask = mask.int()
        loss = []
        for i in range(12):
            input_ = input[i][mask[:, i]==1]
            target_ = target[mask[:, i]==1][:, i]
            if input_.shape[0] > 0:
                loss.append(self.losses[i](input_, target_))
        return loss
    
class MyModel(nn.Module):
    def __init__(self, bert, experts=12, num_tasks=12, dims=2048) -> None:
        super().__init__()
        self.bert = bert
        # self.dense1 = nn.Linear(dims, 768)  # 图像特征降维
        # self.imgconv = nn.Sequential(nn.Conv1d(1, 256, 32, stride=3), nn.BatchNorm1d(256), nn.LeakyReLU(), 
        #                              nn.MaxPool1d(5, stride=5), 
        #                              nn.Conv1d(256, 256, 5, stride=3), nn.BatchNorm1d(256), nn.LeakyReLU(), 
        #                              nn.AdaptiveAvgPool1d(1),
        #                              nn.Flatten(-2, -1), nn.Linear(256, 256))
        # self.imgfuse = nn.Linear(1024, 768)
        self.imgprocess = nn.Linear(dims, 768, bias=False)  # 图像特征变换用于匹配文本特征
        self.mmoe = MMoE(experts, num_tasks)
        self.dense2 = nn.Sequential(nn.Linear(768*2, 512), nn.LeakyReLU())
        self.imgtextmatch = nn.Sequential(nn.Linear(512, 128), nn.LeakyReLU(), 
                                    nn.Linear(128, 1))
        
        self.textprocess = nn.Linear(768, 128)
        self.attmatch = nn.Sequential(nn.Linear(512, 128), nn.LeakyReLU(), 
                                          nn.Linear(128, 12))
        
        self.attrloss = ClassifyLoss()
        self.loss = nn.BCEWithLogitsLoss()
        # weight = torch.tensor([2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2], dtype=torch.float, device='cuda')
        # self.loss = nn.MultiLabelSoftMarginLoss(reduction='sum', weight=weight)

        # weight = torch.tensor([2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2], dtype=torch.float, device='cuda')
        # self.loss = nn.MultiLabelSoftMarginLoss(reduction='sum', weight=weight)
    def attmatch(self, img, text):
        text = self.textprocess(text)
        text_img = torch.cat([img, text], dim=-1)
        
    # def imgprocess(self, img):
    #     # img 信息提取
    #     img_mlp = self.dense1(img)
    #     img_cv = self.imgconv(img.unsqueeze(dim=1))
    #     img = self.imgfuse(torch.cat([img_mlp, img_cv], dim=-1))
    #     return img
        
    def forward(self, input):
        img, text_ids, text_mask, label_attr, mask, neg_text_ids, neg_text_mask, neg_tasks_mask, pos_attr_text_ids, pos_attr_text_mask, pos_tasks_mask = input
        mask = mask.float()
        text_pos = self.bert(text_ids, text_mask)[0][:, 0, :]  
        text_neg = self.bert(neg_text_ids, neg_text_mask)[0][:, 0, :]
        text_attr_pos = self.bert(pos_attr_text_ids, pos_attr_text_mask)[0][:, 0, :]
                
        img = self.imgprocess(img)

        pos_sample = torch.cat([text_pos, img], dim=-1)
        neg_sample = torch.cat([text_neg, img], dim=-1)
        pos_attr_sample = torch.cat([text_attr_pos, img], dim=-1)

        pos_sample = self.dense2(pos_sample)
        neg_sample = self.dense2(neg_sample)
        pos_attr_sample = self.dense2(pos_attr_sample)
                
        pos_img_text_match = self.imgtextmatch(pos_sample)
        neg_img_text_match = self.imgtextmatch(neg_sample)
        pos_attr_img_text_match = self.imgtextmatch(pos_attr_sample)
        
        # pos_attr_match = self.attmatch(pos_sample) # bsz, 12
        # pos_attr_match = pos_attr_match[neg_tasks_mask==1].unsqueeze(dim=-1) # bsz, 1
        
        # 相当于困难样本
        pos_attr_match_hard = self.attmatch(pos_attr_sample) # bsz, 12
        pos_attr_match_hard = pos_attr_match_hard[pos_tasks_mask==1].unsqueeze(dim=-1) # bsz, 1
        
        # attr_hard_label = torch.ones_like(pos_attr_match_hard)
        # loss_attr_hard = self.loss(pos_attr_match_hard, attr_hard_label)
        # if loss_attr_hard.item() < 1e-12:
        #     print(pos_attr_match_hard, attr_hard_label)
        
        neg_attr_match = self.attmatch(neg_sample)
        neg_attr_match = neg_attr_match[neg_tasks_mask==1].unsqueeze(dim=-1)
        
        # neg_attr_hard_label = torch.zeros_like(neg_attr_match)
        # neg_loss_attr_hard = self.loss(neg_attr_match, neg_attr_hard_label)
        
        pred_attr = torch.cat([pos_attr_match_hard, neg_attr_match], dim=-1)
        label_attr = torch.cat([torch.ones_like(pos_attr_match_hard), torch.zeros_like(neg_attr_match)], dim=-1)
        attrloss = self.loss(pred_attr, label_attr)
        # if neg_loss_attr_hard.item() < 1e-12:
        #     print(neg_attr_match, neg_attr_hard_label)
        
        # label_aux = torch.cat([torch.ones_like(pos_attr_match), torch.zeros_like(neg_attr_match)
        #                  ], dim=-1)
        label = torch.cat([torch.ones(img.shape[0], 1, device='cuda'), torch.zeros(img.shape[0], 1, device='cuda')
                    ], dim=-1)
             
        pred = torch.cat([pos_img_text_match, neg_img_text_match], dim=-1)
        
        pred_att_aux = torch.cat([pos_img_text_match, pos_attr_img_text_match], dim=-1)
        
        label_att_aux = self.loss(pred_att_aux, label)
        
        # pred_aux = torch.cat([pos_attr_match, neg_attr_match], dim=-1)
        
        imgtextloss = self.loss(pred, label)
        
        # attr_aux_loss = self.loss(pred_aux, label_aux) 
        
        # attr_out = self.mmoe(img)
        # attrloss = self.attrLoss(attr_out, label_attr, mask)
        
        aux_loss = self.imgTextLoss(img, text_pos)
        
        # loss = imgtextloss / imgtextloss.detach().item() + attrloss / attrloss.detach().item() + \
        #     aux_loss / aux_loss.detach().item() * 0.3 + attr_aux_loss / attr_aux_loss.detach() * 0.4
        # loss = imgtextloss / imgtextloss.detach().item()  + aux_loss / aux_loss.detach().item() * 0.3 + \
        #    attrloss / attrloss.detach().item() + label_att_aux / label_att_aux.detach().item() * 0.5
        
        loss = label_att_aux + attrloss
        return loss
    
    def getAttrScore(self, img):
        attr_out = self.mmoe(img)
        scores = []
        for i in range(len(attr_out)):
            out = attr_out[i]
            scores.append(out)
        return scores
    
    @torch.no_grad()
    def getSubmit(self, input):
        img, text_ids, text_mask = input
        img = self.imgprocess(img)
        # img_norm = F.normalize(img, dim=-1)
        text = self.bert(text_ids, text_mask)[0][:, 0, :]
        sample = torch.cat([text, img], dim=-1)
        sample = self.dense2(sample)
        
        img_text_match_score = F.sigmoid(self.imgtextmatch(sample)).squeeze().cpu().numpy()
        attrscore = self.getAttrScore(img)
        for i in range(len(attrscore)):
            attrscore[i] = attrscore[i].argmax(dim=-1)
            
        attrscore = torch.stack(attrscore, dim=0).permute(1, 0).cpu().numpy() # bsz, num_task
        return img_text_match_score, attrscore
        
    def getAttrLabel(self, img):
        attr_out = self.mmoe(img)
        pred_labels = []
        for i in range(len(attr_out)):
            out = attr_out[i].argmax(dim=-1)
            pred_labels.append(out)
        return pred_labels
    
    def imgTextLoss(self, img, text, all_attr_match=0):
        # 对比损失
        img = F.normalize(img, dim=-1)
        text = F.normalize(text, dim=-1)
        scores = torch.matmul(img, text.t())
        # all_attr_match = torch.diag_embed(all_attr_match) + torch.ones_like(scores, device='cuda')
        # scores = scores * all_attr_match
        pos_scores = torch.diag(scores)
        scores_trans = (scores - pos_scores.unsqueeze(dim=-1)) 
        mask = (scores_trans < -0.2).float() * -1e12
        scores_trans = scores_trans*10 + mask 
        loss = torch.logsumexp(scores_trans, dim=1).mean() + torch.logsumexp(scores_trans, dim=0).mean()
        # pos_score = torch.sum(img * text_pos, dim=-1, keepdim=True)
        # neg_score = torch.sum(img * text_neg, dim=-1, keepdim=True)
        # scores_trans = torch.cat([pos_score, neg_score], dim=-1) - pos_score
        # #scores_trans = scores_trans * 5
        # loss = torch.logsumexp(scores_trans, dim=1).mean()
        return loss
    
    def attrLoss(self, attr_out, label_attr, mask):
        loss = self.attrloss(attr_out, label_attr, mask)  # list:   num_tasks 
        loss = torch.stack(loss, dim=0) 
        loss = torch.sum(loss)
        return loss
    
    @torch.no_grad()
    def getMetric(self, input):
        # 返回 trup positive 个数 分任务
        img, text_ids, text_mask, label_attr, mask, neg_text_ids, neg_text_mask, neg_tasks_mask, pos_attr_text_ids, pos_attr_text_mask, pos_tasks_mask = input

        mask = mask.float()
        text_pos = self.bert(text_ids, text_mask)[0][:, 0, :]  
        text_neg = self.bert(neg_text_ids, neg_text_mask)[0][:, 0, :]
        text_attr_pos = self.bert(pos_attr_text_ids, pos_attr_text_mask)[0][:, 0, :]

        img = self.imgprocess(img)

        pos_sample = torch.cat([text_pos, img], dim=-1)
        neg_sample = torch.cat([text_neg, img], dim=-1)
        pos_attr_sample = torch.cat([text_attr_pos, img], dim=-1)

        pos_sample = self.dense2(pos_sample)
        neg_sample = self.dense2(neg_sample)
        pos_attr_sample = self.dense2(pos_attr_sample)

        neg_img_text_match = F.sigmoid(self.imgtextmatch(neg_sample))
        pos_img_text_match = F.sigmoid(self.imgtextmatch(pos_sample))

        acc_match = (torch.sum(pos_img_text_match>0.5).cpu().item() + torch.sum(neg_img_text_match<0.5).cpu().item())/2

        attr_out = self.mmoe(img)  # num_tasks, batch_size , task_category_num (列表)
        tp_attr, pos_num = [], []
        for i in range(len(attr_out)):
            pred = attr_out[i].argmax(dim=-1)  # batch_size
            tp_attr.append(torch.sum((pred == label_attr[:, i]).float()  * mask[:, i]).item())
            pos_num.append(torch.sum(mask[:, i]).item())

        # 辅助任务
        attr_out2 = F.sigmoid(self.attmatch(pos_sample))
        attr_out2 = torch.where(attr_out2>0.5, torch.ones_like(attr_out2, dtype=torch.int64), 
                                torch.zeros_like(attr_out2, dtype=torch.int64))

        # att_label = neg_tasks_mask.long()
        att_label = mask.long()
        tp_attr2 = torch.sum((attr_out2 == att_label).float() * mask, dim=0).tolist()


        neg_att_match = F.sigmoid(self.attmatch(neg_sample))
        neg_att_match = neg_att_match[neg_tasks_mask==1]  #batch_size
        
        neg_attr_tp = torch.sum(neg_att_match<0.5).item()
        
        # hard pos attr
        pos_attr_match = F.sigmoid(self.attmatch(pos_attr_sample))
        pos_attr_match = pos_attr_match[pos_tasks_mask==1]
        pos_attr_tp = torch.sum(neg_att_match>0.5).item()
        
        return acc_match , tp_attr, pos_num, tp_attr2, neg_attr_tp, pos_attr_tp    # tp_attr pos_num  : 各任务的true positive 和 sum positive
    

