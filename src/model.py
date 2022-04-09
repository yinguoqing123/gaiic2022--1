from torch import exp
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoConfig, AutoTokenizer
from torch.utils.data import DataLoader

class MultiTaskLayer(nn.Module):
    def __init__(self, num_tasks=12, tokenizer=None, bert=None, experts=12, decrease_dim=True) -> None:
        super().__init__()
        self.num_tasks = num_tasks
        self.bert = bert
        self.tokenizer = tokenizer
        self.experts = experts
        self.label_nums = [14, 4, 3, 2, 3, 2, 4, 7, 4, 3, 6, 2]
        self.imgdecrease = nn.ModuleList([nn.Linear(768, 256) for _ in range(experts)])
        self.gates = nn.ModuleList([nn.Linear(768, experts) for i in range(num_tasks)])
        # self.imgdecrease = nn.Linear(768, 256)
        self.textdecrease = nn.Linear(768, 256)
        self.task_layer = nn.ModuleList([nn.Sequential(nn.Linear(256*2, 128), nn.ReLU(), nn.Linear(128, 1)) 
                                    for _ in range(num_tasks)])
        self.queries = self.queryToken()
        self.criterion = nn.BCEWithLogitsLoss()
        
    def forward(self, img):
        # input: (img)   output: num_task, 1  分数 
        query_embeddings = []
        for task_index in range(len(self.queries)):
            query_task_embeddings = []
            for val in self.queries[task_index]:
                val = self.bert(val.cuda())[0][:, 0, :]
                # val = val.detach()
                val = self.textdecrease(val)
                query_task_embeddings.append(val)
                
            query_embeddings.append(query_task_embeddings)
            
        img_experts, gate_out = [], []
        for expert_index in range(self.experts):
            img_experts.append(self.imgdecrease[expert_index](img))
            
        for gate_index in range(self.num_tasks):
            gate_out.append(self.gates[gate_index](img))
            
        img_experts = torch.stack(img_experts, dim=0)  # bsz, experts_num, dim
        gate_out = torch.stack(gate_out, dim=0)  # bsz, num_task, experts_num
        gate_out = torch.softmax(gate_out, dim=-1)
        
        img_out = img_experts.unsqueeze(dim=1) * gate_out.unsqueeze(dim=-1)  # bsz, num_task, experts, dim
        img_out = torch.sum(img_out, dim=-2)  # bsz, num_task, dim
        
        out = []
        for i in range(self.num_tasks):
            input = []
            for j in range(self.label_nums[i]):
                input.append(torch.cat([img_out[:, i, :], query_embeddings[i][j].expand_as(img_out[:, i, :])], dim=-1))
            
            input = torch.stack(input, dim=1)  # bsz, label_nums, dims*2
            layer_out = self.task_layer[i](input)  # bsz, label_nums, 1
            layer_out = layer_out.reshape(layer_out.shape[0], -1)
            out.append(layer_out)
            
        return out  #  num_tasks, bsz, num_category
        
    def queryToken(self):
        queries = [['领型高领半高领立领', '领型连帽可脱卸帽', '领型翻领衬衫领POLO领方领娃娃领荷叶领', '领型双层领', '领型西装领', '领型U型领', '领型一字领', '领型围巾领', '领型堆堆领', '领型V领', '领型棒球领', '领型圆领', '领型斜领', '领型亨利领'], ['袖长短袖五分袖', '袖长九分袖长袖', '袖长七分袖', '袖长无袖'], ['衣长超短款短款常规款', '衣长长款超长款', '衣长中长款'], ['版型修身型标准型', '版型宽松型'], ['裙长短裙超短裙', '裙长中裙中长裙', '裙长长裙'], ['穿着方式套头', '穿着方式开衫'], ['类别手提包', '类别单肩包', '类别斜挎包', '类别双肩包'], ['裤型O型裤锥形裤哈伦裤灯笼裤', '裤型铅笔裤直筒裤小脚裤', '裤型工装裤', '裤型紧身裤', '裤型背带裤', '裤型喇叭裤微喇裤', '裤型阔腿裤'], ['裤长短裤', '裤长五分裤', '裤长七分裤', '裤长九分裤长裤'], ['裤门襟松紧', '裤门襟拉链', '裤门襟系带'], ['闭合方式松紧带', '闭合方式拉链', '闭合方式套筒套脚一脚蹬', '闭合方式系带', '闭合方式魔术贴', '闭合方式搭扣'], [' 鞋帮高度高帮中帮', '鞋帮高度低帮']]
        queryids = []   # num_tasks, category
        for task_index in range(len(queries)):
            task_query_encode = []
            for val in queries[task_index]:
                text_encode = self.tokenizer(val, padding=True, truncation=True, max_length=32, return_attention_mask=True)
                ids = text_encode['input_ids']
                task_query_encode.append(torch.tensor([ids]))
            queryids.append(task_query_encode)
        return queryids
    
class MyModel(nn.Module):
    def __init__(self, bert=None, tokenizer=None, num_tasks=12, dims=2048) -> None:
        super().__init__()
        self.bert = bert
        self.tokenizer = tokenizer
        self.num_tasks = num_tasks
        self.imgdense = nn.Linear(dims, 768)  # 图像特征降维
        self.mltlayer = MultiTaskLayer(num_tasks=12, tokenizer=self.tokenizer, bert=self.bert)
        self.imgtextmatch = nn.Sequential(nn.Linear(768*2, 512), nn.LeakyReLU(), nn.Linear(512, 128), nn.LeakyReLU(), 
                                    nn.Linear(128, 1))
        self.criterion = nn.BCEWithLogitsLoss()
        # weight = torch.tensor([2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2], dtype=torch.float, device='cuda')
        # self.loss = nn.MultiLabelSoftMarginLoss(reduction='sum', weight=weight)
        # weight = torch.tensor([2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2], dtype=torch.float, device='cuda')
        # self.loss = nn.MultiLabelSoftMarginLoss(reduction='sum', weight=weight)
        
    def forward(self, input):
        img, text_ids, text_mask, label_attr, mask, neg_text_ids, neg_text_mask = input
        mask = mask.float()
        text_pos = self.bert(text_ids, text_mask)[0][:, 0, :]  
        text_neg = self.bert(neg_text_ids, neg_text_mask)[0][:, 0, :]
                
        img = self.imgdense(img)
        pos_sample = torch.cat([text_pos, img], dim=-1)
        neg_sample = torch.cat([text_neg, img], dim=-1)

        pos_img_text_match = self.imgtextmatch(pos_sample)
        neg_img_text_match = self.imgtextmatch(neg_sample)

        label = torch.cat([torch.ones_like(pos_img_text_match), torch.zeros_like(neg_img_text_match)
                            ], dim=-1)     
        pred = torch.cat([pos_img_text_match, neg_img_text_match], dim=-1)
        imgtextloss = self.criterion(pred, label)
        
        attr_out = self.mltlayer(img)   # 列表 nums_task,  bsz, task_category  
        attrloss = self.attrLoss(attr_out, label_attr, mask)
        
        loss = imgtextloss + attrloss
        return loss
    
    @torch.no_grad()
    def getSubmit(self, input):
        img, text_ids, text_mask = input
        img = self.imgdense(img)
        
        text = self.bert(text_ids, text_mask)[0][:, 0, :]
        sample = torch.cat([text, img], dim=-1)
        
        img_text_match_score = F.sigmoid(self.imgtextmatch(sample)).squeeze().cpu().numpy()
        
        attrscore = self.mltlayer(img)  # num_tasks, bsz, num_category
        for i in range(len(attrscore)):
            attrscore[i] = attrscore[i].argmax(dim=-1)
        attrscore = torch.stack(attrscore, dim=0).permute(1, 0).cpu().numpy() # bsz, num_task
        return img_text_match_score, attrscore
 
    def attrLoss(self, attr_out, label_attr, mask):
        # attr_out: 列表 nums_task,  bsz, task_category
        loss = []
        for i in range(len(attr_out)):
            attr_label = label_attr[:, i]
            attr_label = F.one_hot(attr_label, num_classes=attr_out[i].shape[-1])
            if torch.sum(mask[:, i]) > 0:
                loss.append( self.criterion(attr_out[i][mask[:, i]==1], attr_label[mask[:, i]==1].float()) )
        
        loss = torch.stack(loss, dim=0) 
        loss = torch.sum(loss)
        return loss
    
    @torch.no_grad()
    def getMetric(self, input):
        # 返回 trup positive 个数 分任务
        img, text_ids, text_mask, label_attr, mask, neg_text_ids, neg_text_mask = input

        mask = mask.float()
        text_pos = self.bert(text_ids, text_mask)[0][:, 0, :]  
        text_neg = self.bert(neg_text_ids, neg_text_mask)[0][:, 0, :]

        img = self.imgdense(img)

        pos_sample = torch.cat([text_pos, img], dim=-1)
        neg_sample = torch.cat([text_neg, img], dim=-1)

        neg_img_text_match = F.sigmoid(self.imgtextmatch(neg_sample))
        pos_img_text_match = F.sigmoid(self.imgtextmatch(pos_sample))

        acc_match = (torch.sum(pos_img_text_match>0.5).cpu().item() + torch.sum(neg_img_text_match<0.5).cpu().item())/2

        attr_out = self.mltlayer(img)  # num_tasks, batch_size , task_category_num (列表)
        
        tp_attr, pos_num = [], []
        for i in range(len(attr_out)):
            pred = attr_out[i].argmax(dim=-1)  # batch_size
            tp_attr.append(torch.sum((pred == label_attr[:, i]).float()  * mask[:, i]).item())
            pos_num.append(torch.sum(mask[:, i]).item())
            
        return acc_match , tp_attr, pos_num    # tp_attr pos_num  : 各任务的true positive 和 sum positive
    

