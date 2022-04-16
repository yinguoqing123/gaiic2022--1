import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoConfig, AutoTokenizer
from torch.utils.data import DataLoader
from utils import FocalLoss

class MMoE(nn.Module):
    def __init__(self, num_experts=10, num_tasks=12, dims=768) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_spe = 1
        self.num_tasks = num_tasks
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(dims, 256), nn.ReLU()) 
                                      for i in range(num_experts)])
        self.experts_spe = nn.ModuleDict({f'task_{task_index}': nn.ModuleList([nn.Sequential(nn.Linear(dims, 256), nn.ReLU()) for _ in range(self.num_experts_spe)]) for task_index in range(self.num_tasks)})
        # self.gates = nn.ModuleList([nn.Sequential(nn.Linear(dims, 128), nn.ReLU(), nn.Linear(128, self.num_experts+self.num_experts_spe)) for i in range(num_tasks)])
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(dims, self.num_experts+self.num_experts_spe)) for i in range(num_tasks)])
        self.label_nums = [14, 4, 3, 2, 3, 2, 4, 7, 4, 3, 6, 2]
        self.classifies = nn.ModuleList([nn.Linear(256, num) for num in self.label_nums])
        
    def forward(self, input):
        out_experts = []
        for layer in self.experts:
            out_experts.append(layer(input))  # num_experts, b, 128

        out_experts = torch.stack(out_experts, dim=0).permute(1, 0, 2) # b, num_experts, 128
        
        out_experts_spe = []
        for task_index in range(self.num_tasks):
            for layer in self.experts_spe[f'task_{task_index}']:
                out_experts_spe.append(layer(input))   # num_task * num_experts_spe, bsz, dim
                
        out_experts_spe = torch.stack(out_experts_spe, dim=0).reshape(self.num_tasks, self.num_experts_spe, -1, 256).permute(2, 0, 1, 3)  #  bsz, num_task, num_experts_spe, dim
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
                
        loss = torch.stack(loss, dim=0)
        loss = torch.sum(loss)
        return loss
    
class MyModel(nn.Module):
    def __init__(self, bert, experts=10, num_tasks=12, dims=2048) -> None:
        super().__init__()
        self.bert = bert
        # self.imgprocess = nn.Linear(dims, 768, bias=False)  # 图像特征变换用于匹配文本特征
        self.imgprocess = nn.Sequential(nn.Linear(2048, 1024), nn.LeakyReLU(), nn.Linear(1024, 768))
        self.imgtextmatch = nn.Sequential(nn.Linear(768, 512), nn.LeakyReLU(), nn.Linear(512, 1))
        self.mmoe = MMoE(experts, num_tasks)  
        self.attrloss = ClassifyLoss()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, input):
        img, text_ids, text_mask, label_attr, mask, neg_text_ids, neg_text_mask, neg_tasks_mask, \
            pos_attr_text_ids, pos_attr_text_mask, pos_tasks_mask, pos_title_mask, neg_title_mask = input
        mask = mask.float()
        img = self.imgprocess(img)
        text_mask = torch.cat([text_mask, torch.ones(text_mask.shape[0], 1, device='cuda', dtype=torch.int64)], dim=-1)
        text_pos = self.bert(text_ids, text_mask, visual_embeds=img)[0][:, 0, :] 
        neg_text_mask = torch.cat([neg_text_mask, torch.ones(neg_text_mask.shape[0], 1, device='cuda', dtype=torch.int64)], dim=-1) 
        text_neg = self.bert(neg_text_ids, neg_text_mask, visual_embeds=img)[0][:, 0, :]                

        # pos_sample = torch.cat([text_pos, img], dim=-1)
        # neg_sample = torch.cat([text_neg, img], dim=-1)
        
        pos_sample = text_pos
        neg_sample = text_neg

        pos_sample = self.imgtextmatch(pos_sample).squeeze()   # bsz, 13
        neg_sample = self.imgtextmatch(neg_sample).squeeze()   # bsz, 13

        pos_sample[pos_title_mask==0] = 1e12
        neg_sample[neg_title_mask==0] = -1e12

        label_imgtext = torch.cat([torch.ones(pos_sample.shape[0],  1, device='cuda'), torch.zeros(pos_sample.shape[0], 1, device='cuda')], dim=-1)  # bsz, 3     
        pred_imgtext = torch.stack([pos_sample, neg_sample], dim=-1)  # bsz, 2
        loss_imgtext = self.loss(pred_imgtext, label_imgtext)

        attr_pred = self.mmoe(img) 
        loss_attr = self.attrloss(attr_pred, label_attr, mask)  
        
        aux_loss = self.imgTextLoss(img, text_pos) 

        loss = torch.log(loss_imgtext) + torch.log(max(loss_attr, torch.tensor(1e-12, device='cuda'))) + torch.log(max(aux_loss, torch.tensor(1e-10, device='cuda'))) * 0.2
        return loss
    
    def getAttrScore(self, img):
        attr_out = self.mmoe(img)
        scores = []
        for i in range(len(attr_out)):
            out = torch.softmax(attr_out[i], dim=-1)
            scores.append(out)
        return scores
    
    @torch.no_grad()
    def getSubmit(self, input):
        img, text_ids, text_mask, label_attr = input
        img = self.imgprocess(img)
        text_mask = torch.cat([text_mask, torch.ones(text_mask.shape[0], 1, device='cuda', dtype=torch.int64)], dim=-1)
        text = self.bert(text_ids, text_mask, visual_embeds=img)[0][:, 0, :]
        # sample =  torch.cat([text, img], dim=-1)
        sample = F.sigmoid(self.imgtextmatch(text).squeeze())
        img_text_match_score = sample.cpu().numpy()
        
        attrscore = self.getAttrScore(img)
        for i in range(len(attrscore)):
            attrscore[i] = torch.gather(attrscore[i], 1, label_attr[:, i].view(-1, 1)).squeeze()
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
        img, text_ids, text_mask, label_attr, mask, neg_text_ids, neg_text_mask, neg_tasks_mask, pos_attr_text_ids, \
            pos_attr_text_mask, pos_tasks_mask, pos_title_mask, neg_title_mask  = input

        mask = mask.float()

        img = self.imgprocess(img)
        text_mask = torch.cat([text_mask, torch.ones(text_mask.shape[0], 1, device='cuda', dtype=torch.int64)], dim=-1)
        text_pos = self.bert(text_ids, text_mask, visual_embeds=img)[0][:, 0, :] 
        neg_text_mask = torch.cat([neg_text_mask, torch.ones(neg_text_mask.shape[0], 1, device='cuda', dtype=torch.int64)], dim=-1) 
        text_neg = self.bert(neg_text_ids, neg_text_mask, visual_embeds=img)[0][:, 0, :]
    
        # pos_sample = torch.cat([text_pos, img], dim=-1)
        # neg_sample = torch.cat([text_neg, img], dim=-1)
        
        pos_sample = text_pos
        neg_sample = text_neg
                
        pos_sample = F.sigmoid(self.imgtextmatch(pos_sample).squeeze())
        neg_sample = F.sigmoid(self.imgtextmatch(neg_sample).squeeze())

        pos_img_text_match = pos_sample
        neg_img_text_match = neg_sample

        pos_img_text_match = pos_img_text_match[pos_title_mask==1]
        neg_img_text_dual_match = neg_img_text_match[(neg_title_mask==1)&(torch.sum(mask, dim=1)>=1)]
        neg_img_text_match = neg_img_text_match[(neg_title_mask==1)&(torch.sum(mask, dim=1)<1)]

        acc_match_pos =  torch.sum(pos_img_text_match>0.5).cpu().item() 
        acc_match_dual_neg = torch.sum(neg_img_text_dual_match<0.5).cpu().item() 
        acc_match_neg = torch.sum(neg_img_text_match<0.5).cpu().item() 

        attr_out = self.mmoe(img)  # num_tasks, batch_size , task_category_num (列表)
        tp_attr, pos_num = [], []
        for i in range(len(attr_out)):
            pred = attr_out[i].argmax(dim=-1)  # batch_size
            tp_attr.append(torch.sum((pred == label_attr[:, i]).float()  * mask[:, i]).item())
            pos_num.append(torch.sum(mask[:, i]).item())

        
        return acc_match_pos, acc_match_dual_neg, acc_match_neg ,  tp_attr, pos_num  # tp_attr pos_num  : 各任务的true positive 和 sum positive
    

