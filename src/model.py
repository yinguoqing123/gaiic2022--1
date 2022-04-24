import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoConfig, AutoTokenizer
from torch.utils.data import DataLoader
from utils import FocalLoss, get_coef

class MyModel(nn.Module):
    def __init__(self, bert, experts=10, num_tasks=12, dims=2048) -> None:
        super().__init__()
        self.bert = bert
        self.imgprocess = nn.Sequential(nn.Linear(2048, 1024), nn.LeakyReLU(), nn.Linear(1024, 768), nn.Dropout(0.1))
        self.match = nn.Sequential(nn.Linear(768, 512), nn.LeakyReLU(), nn.Linear(512, 13))
        # self.match1 = nn.Linear(512, 1)
        # self.match2 = nn.Sequential(nn.Linear(512, 128), nn.LeakyReLU(), nn.Linear(128, 12))
        self.loss = nn.BCEWithLogitsLoss()
        self.coef = torch.tensor(get_coef(13), device='cuda', dtype=torch.float) # 4096*13
        self.loss_cross = nn.CrossEntropyLoss()

    def forward(self, input):
        img, text_ids, text_mask, label_attr, mask, neg_text_ids, neg_text_mask, neg_tasks_mask, \
            pos_attr_text_ids, pos_attr_text_mask, pos_tasks_mask, pos_title_mask, neg_title_mask = input
        mask = mask.float()
        img = self.imgprocess(img)
        text_mask = torch.cat([text_mask, torch.ones(text_mask.shape[0], 1, device='cuda', dtype=torch.int64)], dim=-1)
        text_pos = self.bert(text_ids, text_mask, visual_embeds=img)[0][:, 0, :] 
        neg_text_mask = torch.cat([neg_text_mask, torch.ones(neg_text_mask.shape[0], 1, device='cuda', dtype=torch.int64)], dim=-1) 
        text_neg = self.bert(neg_text_ids, neg_text_mask, visual_embeds=img)[0][:, 0, :]                
        pos_attr_text_mask = torch.cat([pos_attr_text_mask, torch.ones(pos_attr_text_mask.shape[0], 1, device='cuda', dtype=torch.int64)], dim=-1)
        text_attr_pos = self.bert(pos_attr_text_ids, pos_attr_text_mask, visual_embeds=img)[0][:, 0, :]
            
        # pos_sample = torch.cat([text_pos, img], dim=-1)
        # neg_sample = torch.cat([text_neg, img], dim=-1)
        # pos_attr_sample = torch.cat([text_attr_pos, img], dim=-1)

        pos_sample = text_pos
        neg_sample = text_neg
        pos_attr_sample = text_attr_pos

        pos_sample = self.match(pos_sample)   # bsz, 13
        neg_sample = self.match(neg_sample)   # bsz, 13
        pos_attr_sample = self.match(pos_attr_sample)[:, 1:]

        neg_sample[:, 0] = neg_sample[:, 0] * -1.0
        neg_sample[:, 1:] = torch.where((mask==1)&(neg_tasks_mask==0), neg_sample[:, 1:]*-1.0, neg_sample[:, 1:])

        pos_sample[:, 0][pos_title_mask==0] = 0
        pos_sample[:, 1:][mask==0] = 0

        neg_sample[:, 0][neg_title_mask==0] = 0
        neg_sample[:, 1:][mask==0] = 0

        pos_sample = torch.matmul(self.coef.unsqueeze(dim=0), pos_sample.unsqueeze(dim=-1)).squeeze() # bsz, 4096
        neg_sample = torch.matmul(self.coef.unsqueeze(dim=0), neg_sample.unsqueeze(dim=-1)).squeeze() # bsz, 4096
                
        pred_img_text = torch.cat([pos_sample, neg_sample], dim=0)
        label_img_text = torch.zeros_like(pred_img_text[:, 0], dtype=torch.long)
        loss_imgtext = self.loss_cross(pred_img_text, label_img_text)

        # pos_sample[:, 0][pos_title_mask==0] = 1e12
        # neg_sample[:, 0][neg_title_mask==0] = -1e12
        

        # label_imgtext = torch.cat([torch.ones(pos_sample.shape[0],  1, device='cuda'), torch.zeros(pos_sample.shape[0], 1, device='cuda')], dim=-1)  # bsz, 3     
        # pred_imgtext = torch.stack([pos_sample[:, 0], neg_sample[:, 0]], dim=-1)  # bsz, 2
        # loss_imgtext = self.loss(pred_imgtext, label_imgtext)

        pos_attr_sample[mask==0] = -1e12
        loss_attr = self.loss(pos_attr_sample, pos_tasks_mask.float())
        loss = torch.log(loss_imgtext) + torch.log(loss_attr) 
        
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
        sample = text
        sample = F.sigmoid(self.match(sample).squeeze())
        img_text_match_score = sample[:, 0].cpu().numpy()
        attrscore = sample[:, 1:].cpu().numpy() 
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
        pos_attr_text_mask = torch.cat([pos_attr_text_mask, torch.ones(pos_attr_text_mask.shape[0], 1, device='cuda', dtype=torch.int64)], dim=-1)
        text_attr_pos = self.bert(pos_attr_text_ids, pos_attr_text_mask, visual_embeds=img)[0][:, 0, :]


        pos_sample = text_pos
        neg_sample = text_neg
        pos_attr_sample = text_attr_pos       

        pos_sample = F.sigmoid(self.match(pos_sample))
        neg_sample = F.sigmoid(self.match(neg_sample))
        pos_attr_sample = F.sigmoid(self.match(pos_attr_sample))

        pos_img_text_match = pos_sample[:, 0]
        neg_img_text_match = neg_sample[:, 0]

        pos_img_text_match = pos_img_text_match[pos_title_mask==1]
        neg_img_text_dual_match = neg_img_text_match[(neg_title_mask==1)&(pos_title_mask==1)]
        neg_img_text_match = neg_img_text_match[(neg_title_mask==1)&(pos_title_mask==0)]

        # acc_match_pos =  torch.sum(pos_img_text_match>0.5).cpu().item() 
        acc_match_pos =  pos_img_text_match.cpu().numpy()
        # acc_match_dual_neg = torch.sum(neg_img_text_dual_match<0.5).cpu().item() 
        acc_match_dual_neg = neg_img_text_dual_match.cpu().numpy() 
        # acc_match_neg = torch.sum(neg_img_text_match<0.5).cpu().item() 
        acc_match_neg = neg_img_text_match.cpu().numpy() 


        pos_attr_attr_match = pos_attr_sample[pos_title_mask==1][:, 1:] # bsz, 12

        pos_attr_attr_match = torch.where(pos_attr_attr_match>0.5, torch.ones_like(pos_attr_attr_match, dtype=torch.int64), 
                                torch.zeros_like(pos_attr_attr_match, dtype=torch.int64))

        tp_pos_attr_attr = torch.sum((pos_attr_attr_match.long() == pos_tasks_mask[pos_title_mask==1].long()).float() * mask[pos_title_mask==1], dim=0).tolist()

        pos_num = torch.sum(mask[pos_title_mask==1], dim=0).tolist()
        return acc_match_pos, acc_match_dual_neg, acc_match_neg ,  tp_pos_attr_attr, pos_num,  # tp_attr pos_num  : 各任务的true positive 和 sum positive
       
