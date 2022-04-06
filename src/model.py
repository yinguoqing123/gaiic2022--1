import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoConfig, AutoTokenizer
from torch.utils.data import DataLoader

class MMoE(nn.Module):
    def __init__(self, num_experts=12, num_tasks=12, dims=768) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(dims, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU()) 
                                      for i in range(num_experts)])
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(dims, 128), nn.ReLU(), nn.Linear(128, num_experts)) for i in range(num_tasks)])
        self.label_nums = [14, 4, 3, 2, 3, 2, 4, 7, 4, 3, 6, 2]
        self.classifies = nn.ModuleList([nn.Linear(128, num) for num in self.label_nums])
        
    def forward(self, input):
        out_experts = []
        for layer in self.experts:
            out_experts.append(layer(input))  # num_experts, b, 128

        out_experts = torch.stack(out_experts, dim=0).permute(1, 0, 2) # b, num_experts, 128
            
        out_gates = []
        for layer in self.gates:
            out_gates.append(layer(input))  # num_tasks, b, num_experts
            
        out_gates = torch.stack(out_gates, dim=0).permute(1, 0, 2)  # b, num_tasks, num_experts
        out_gates = torch.softmax(out_gates, dim=-1)

        out = out_experts.unsqueeze(dim=1) * out_gates.unsqueeze(dim=-1)  # b, num_tasks, num_experts, 128
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
        self.dense1 = nn.Linear(dims, 1024)  # 图像特征降维
        self.dense2 = nn.Linear(1024, 768, bias=False)  # 图像特征变换用于匹配文本特征
        self.mmoe = MMoE(experts, num_tasks)
        self.attrloss = ClassifyLoss()
        
    def forward(self, input):
        img, text_ids, text_mask, label_attr, mask = input
        mask = mask.float()
        text = self.bert(text_ids, text_mask)[0][:, 0, :]
        img = self.dense1(img)
        img = self.dense2(torch.relu(img))
        
        attr_out = self.mmoe(img)  # num_tasks, batch_size , task_category_num (列表)
        
        attr_out_prob = torch.stack([F.pad(batch_task, (0, 14-batch_task.shape[1])) for batch_task in attr_out])
        attr_out_prob = attr_out_prob.permute(1, 0, 2)  # bsz, num_tasks, 14  最大的类别数是14
        label_attr_trans = F.one_hot(label_attr, num_classes=14) # bsz, num_tasks, 14
        all_attr_match = attr_out_prob * mask.unsqueeze(dim=-1).float() * label_attr_trans.float()  # bsz, num_task, 14,   mask、one_hot都为1才有效
        fillval = torch.ones_like(all_attr_match, device='cuda')
        all_attr_match = torch.where(all_attr_match<1e-12, fillval, all_attr_match)
        all_attr_match = torch.prod(all_attr_match, dim=-1)  # bsz, num_task
        all_attr_match = torch.prod(all_attr_match, dim=-1)  # bsz
        
        text_img_loss = self.imgTextLoss(img, text, all_attr_match)  # 图文匹配得分
        
        attr_loss = self.attrLoss(attr_out, label_attr, mask)
        loss = text_img_loss + attr_loss
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
        img_norm = F.normalize(img, dim=-1)
        text = self.bert(text_ids, text_mask)[0][:, 0, :] 
        text_norm = F.normalize(text, dim=-1)
        imgtextscore = torch.sum(img_norm*text_norm, dim=-1).cpu().numpy()  # bsz
        attrscore = self.getAttrScore(img)
        for i in range(len(attrscore)):
            attrscore[i] = attrscore[i].argmax(dim=-1)
        attrscore = torch.stack(attrscore, dim=0).permute(1, 0).cpu().numpy() # bsz, num_task
        return imgtextscore, attrscore
        
    def getAttrLabel(self, img):
        attr_out = self.mmoe(img)
        pred_labels = []
        for i in range(len(attr_out)):
            out = attr_out[i].argmax(dim=-1)
            pred_labels.append(out)
        return pred_labels
    
    def imgTextLoss(self, img, text, all_attr_match):
        # 对比损失
        img = F.normalize(img, dim=-1)
        text = F.normalize(text, dim=-1)
        scores = torch.matmul(img, text.t())
        all_attr_match = torch.diag_embed(all_attr_match) + torch.ones_like(scores, device='cuda')
        scores = scores * all_attr_match
        pos_scores = torch.diag(scores)
        scores_trans = (scores - pos_scores.unsqueeze(dim=-1)) * 20
        loss = torch.logsumexp(scores_trans, dim=1).mean() + torch.logsumexp(scores_trans, dim=0).mean()
        return loss
    
    def attrLoss(self, attr_out, label_attr, mask):
        loss = self.attrloss(attr_out, label_attr, mask)  # list:   num_tasks 
        loss = torch.stack(loss, dim=0) 
        loss = torch.sum(loss)
        return loss
    
    def getMetric(self, input):
        # 返回 trup positive 个数 分任务
        img, text_ids, text_mask, label_attr, mask = input
        mask = mask.float()
        text = self.bert(text_ids, text_mask)[0][:, 0, :]
        img = self.dense1(img)
        img = self.dense2(torch.relu(img))
        img_norm = F.normalize(img, dim=-1)
        text_norm = F.normalize(text, dim=-1)
        scores = torch.matmul(img_norm, text_norm.t())  
        indices = scores.argmax(dim=-1)
        acc_match = torch.sum(torch.eq(indices, torch.arange(text.shape[0]).cuda())).item()

        attr_out = self.mmoe(img)  # num_tasks, batch_size , task_category_num (列表)
        tp_attr, pos_num = [], []
        for i in range(len(attr_out)):
            pred = attr_out[i].argmax(dim=-1)  # batch_size
            tp_attr.append(torch.sum((pred == label_attr[:, i]).float()  * mask[:, i]).item())
            pos_num.append(torch.sum(mask[:, i]).item())
            
        return acc_match , tp_attr, pos_num    # tp_attr pos_num  : 各任务的true positive 和 sum positive
    
