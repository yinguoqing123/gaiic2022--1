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
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(dims, self.num_experts+self.num_experts_spe)) for i in range(num_tasks)])
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
    
class AttentionPooling1D(nn.Module):
    """通过加性Attention, 将向量序列融合为一个定长向量
    """
    def __init__(self, in_features,  **kwargs):
        super(AttentionPooling1D, self).__init__(**kwargs)
        self.in_features = in_features # 词向量维度
        self.k_dense = nn.Linear(self.in_features, self.in_features, bias=False)
    def forward(self, xo, query, mask):
        x = self.k_dense(xo)  # x: bsz, seq_len, dim
        query = query.unsqueeze(dim=0).permute(0, 2, 1)  # 1, bsz, dim => 1, dim, bsz
        out = torch.matmul(x, query).permute(0, 2, 1)  #bsz, seq, bsz => bsz, bsz, seq
        mask = mask.unsqueeze(dim=1)   #  bsz, 1,  seq_len 
        out = out - (1-mask.float()) * 1e12  # bsz, bsz, seq_len
        out = F.softmax(out, dim=-1) 
        out = x.unsqueeze(dim=1) * out.unsqueeze(dim=-1) # bsz, bsz, seq, dim
        out = torch.sum(out, dim=-2)  # bsz, bsz, dim
        return out
        # x = torch.sum(x * query.unsqueeze(dim=1), dim=-1)
        # x = x - (1 - mask.float()) * 1e12
        # x = F.softmax(x, dim=-1)  # bsz, seq_len
        #return torch.sum(x.unsqueeze(dim=-1) * xo, dim=-2) #bsz, bsz, dim

    
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
        self.dense1 = nn.Linear(dims, 768)  # 图像特征降维
        # self.dense2 = nn.Linear(1024, 768, bias=False)  # 图像特征变换用于匹配文本特征
        # self.mmoe = MMoE(experts, num_tasks)
        self.dense2 = nn.Sequential(nn.Linear(768*2, 512), nn.LeakyReLU())
        self.imgtextmatch = nn.Sequential(nn.Linear(512, 128), nn.LeakyReLU(), 
                                          nn.Linear(128, 1))
        self.attmatch = nn.Sequential(nn.Linear(512, 128), nn.LeakyReLU(), 
                                          nn.Linear(128, 12))
        # self.attrloss = ClassifyLoss()
        self.loss = nn.MultiLabelSoftMarginLoss(reduction='sum')

        
    def forward(self, input):
        img, text_ids, text_mask, label_attr, mask, neg_text_ids, neg_text_mask, neg_tasks_mask = input
        mask = mask.float()
        text_pos = self.bert(text_ids, text_mask)[0][:, 0, :]  
        text_neg = self.bert(neg_text_ids, neg_text_mask)[0][:, 0, :]
                
        img = self.dense1(img)

        pos_sample = torch.cat([text_pos, img], dim=-1)
        neg_sample = torch.cat([text_neg, img], dim=-1)

        pos_sample = self.dense2(pos_sample)
        neg_sample = self.dense2(neg_sample)
                
        pos_img_text_match = self.imgtextmatch(pos_sample)
        neg_img_text_match = self.imgtextmatch(neg_sample)
                
                # pos_attr_match = self.attmatch(pos_sample)
        neg_att_match = self.attmatch(neg_sample)
        pos_att_match = self.attmatch(pos_sample)
        
        neg_att_match[mask==0] = -1e12
        pos_att_match[mask==0] = -1e12
                
        label = torch.cat([torch.ones(img.shape[0], 1, device='cuda'), torch.zeros(img.shape[0], 1, device='cuda'),
                            neg_tasks_mask.float(), mask], dim=-1)
                
        pred = torch.cat([pos_img_text_match, neg_img_text_match, neg_att_match, pos_att_match], dim=-1)
        loss = self.loss(pred, label)
                
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
        img = self.dense1(img)
        # img_norm = F.normalize(img, dim=-1)
        text = self.bert(text_ids, text_mask)[0][:, 0, :]
        sample = torch.cat([text, img], dim=-1)
        sample = self.dense2(sample)
        
        img_text_match_score = F.sigmoid(self.imgtextmatch(sample)).squeeze().cpu().numpy()
        att_match_score = F.sigmoid(self.attmatch(sample)).cpu().numpy()

        return img_text_match_score, att_match_score
        
    def getAttrLabel(self, img):
        attr_out = self.mmoe(img)
        pred_labels = []
        for i in range(len(attr_out)):
            out = attr_out[i].argmax(dim=-1)
            pred_labels.append(out)
        return pred_labels
    
    def imgTextLoss(self, img, text_pos, text_neg, all_attr_match):
        # 对比损失
        img = F.normalize(img, dim=-1)
        text_pos = F.normalize(text_pos, dim=-1)
        text_neg = F.normalize(text_neg, dim=-1)
        # scores = torch.matmul(img, text.t())
        # all_attr_match = torch.diag_embed(all_attr_match) + torch.ones_like(scores, device='cuda')
        # scores = scores * all_attr_match
        # pos_scores = torch.diag(scores)
        # scores_trans = (scores - pos_scores.unsqueeze(dim=-1)) 
        # mask = (scores_trans < -0.2).float() * -1e12
        # scores_trans = scores_trans*20 + mask 
        # loss = torch.logsumexp(scores_trans, dim=1).mean() + torch.logsumexp(scores_trans, dim=0).mean()
        pos_score = torch.sum(img * text_pos, dim=-1, keepdim=True)
        neg_score = torch.sum(img * text_neg, dim=-1, keepdim=True)
        scores_trans = torch.cat([pos_score, neg_score], dim=-1) - pos_score
        #scores_trans = scores_trans * 5
        loss = torch.logsumexp(scores_trans, dim=1).mean()
        return loss
    
    def attrLoss(self, attr_out, label_attr, mask):
        loss = self.attrloss(attr_out, label_attr, mask)  # list:   num_tasks 
        loss = torch.stack(loss, dim=0) 
        loss = torch.sum(loss)
        return loss
    
    def getMetric(self, input):
        # 返回 trup positive 个数 分任务
        img, text_ids, text_mask, label_attr, mask, neg_text_ids, neg_text_mask, neg_tasks_mask = input
        
        mask = mask.float()
        text_pos = self.bert(text_ids, text_mask)[0][:, 0, :]  
        text_neg = self.bert(neg_text_ids, neg_text_mask)[0][:, 0, :]

        img = self.dense1(img)

        pos_sample = torch.cat([text_pos, img], dim=-1)
        neg_sample = torch.cat([text_neg, img], dim=-1)

        pos_sample = self.dense2(pos_sample)
        neg_sample = self.dense2(neg_sample)

        pos_img_text_match = F.sigmoid(self.imgtextmatch(pos_sample))
        neg_img_text_match = F.sigmoid(self.imgtextmatch(neg_sample))

                # pos_attr_match = self.attmatch(pos_sample)
        neg_att_match = F.sigmoid(self.attmatch(neg_sample))
        pos_att_match = F.sigmoid(self.attmatch(pos_sample))
        neg_att_match = torch.where(neg_att_match>0.5, torch.ones_like(neg_att_match, dtype=torch.int64), torch.zeros_like(neg_att_match, dtype=torch.int64))
        pos_att_match = torch.where(pos_att_match>0.5, torch.ones_like(pos_att_match, dtype=torch.int64), torch.zeros_like(pos_att_match, dtype=torch.int64))             
        
        att_match = torch.cat([neg_att_match, pos_att_match], dim=-1)
        att_label = torch.cat([neg_tasks_mask.float(), mask], dim=-1).long()
        tp_attr = torch.sum((att_match == att_label).float() * torch.cat([mask, mask], dim=-1), dim=0).tolist()
        
        pos_num = torch.sum(mask, dim=0).repeat(2).tolist()
        
        acc_match = torch.sum(pos_img_text_match>0.5).cpu().item() + torch.sum(neg_img_text_match<0.5).cpu().item()

            
        return acc_match , tp_attr, pos_num    # tp_attr pos_num  : 各任务的true positive 和 sum positive
    

