
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from collections import deque

tasks = ['领型', '袖长', '衣长', '版型', '裙长', '穿着方式', '类别', '裤型', '裤长', '裤门襟', 
        '闭合方式', '鞋帮高度']

tasksMap = {'领型': 0, '袖长': 1, '衣长': 2, '版型': 3, '裙长': 4, '穿着方式': 5, '类别': 6, 
            '裤型': 7, '裤长': 8, '裤门襟': 9, '闭合方式': 10, '鞋帮高度': 11}

vals = [['高领=半高领=立领', '连帽=可脱卸帽', '翻领=衬衫领=POLO领=方领=娃娃领=荷叶领', '双层领', 
        '西装领', 'U型领', '一字领', '围巾领', '堆堆领', 'V领', '棒球领', '圆领', '斜领', '亨利领'], 
       ['短袖=五分袖', '九分袖=长袖', '七分袖', '无袖'], ['超短款=短款=常规款', '长款=超长款', '中长款'], 
       ['修身型=标准型', '宽松型'], ['短裙=超短裙', '中裙=中长裙', '长裙'], ['套头', '开衫'], 
       ['手提包', '单肩包', '斜挎包', '双肩包'], 
       ['O型裤=锥形裤=哈伦裤=灯笼裤', '铅笔裤=直筒裤=小脚裤', '工装裤', '紧身裤', '背带裤', 
        '喇叭裤=微喇裤', '阔腿裤'], ['短裤', '五分裤', '七分裤', '九分裤=长裤'], 
       ['松紧', '拉链', '系带'], ['松紧带', '拉链', '套筒=套脚=一脚蹬', '系带', '魔术贴', '搭扣'], 
       ['高帮=中帮', '低帮']]

query = []
for task_name, vals in zip(tasks, vals):
    query_task = []
    for val in vals:
        val = ''.join(val.split('='))
        query_task.append(task_name + val)
    query.append(query_task)

valsMap = [{'高领': 0, '半高领': 0, '立领': 0, '连帽': 1, '可脱卸帽': 1, '翻领': 2, '衬衫领': 2, 'POLO领': 2, '方领': 2, '娃娃领': 2, '荷叶领': 2, '双层领': 3, '西装领': 4, 'U型领': 5, '一字领': 6, '围巾领': 7, '堆堆领': 8, 'V领': 9, '棒球领': 10, '圆领': 11, '斜领': 12, '亨利领': 13}, 
           {'短袖': 0, '五分袖': 0, '九分袖': 1, '长袖': 1, '七分袖': 2, '无袖': 3}, 
           {'超短款': 0, '短款': 0, '常规款': 0, '长款': 1, '超长款': 1, '中长款': 2}, 
           {'修身型': 0, '标准型': 0, '宽松型': 1}, 
           {'短裙': 0, '超短裙': 0, '中裙': 1, '中长裙': 1, '长裙': 2}, 
           {'套头': 0, '开衫': 1}, 
           {'手提包': 0, '单肩包': 1, '斜挎包': 2, '双肩包': 3}, 
           {'O型裤': 0, '锥形裤': 0, '哈伦裤': 0, '灯笼裤': 0, '铅笔裤': 1, '直筒裤': 1, '小脚裤': 1, '工装裤': 2, '紧身裤': 3, '背带裤': 4, '喇叭裤': 5, '微喇裤': 5, '阔腿裤': 6}, 
           {'短裤': 0, '五分裤': 1, '七分裤': 2, '九分裤': 3, '长裤': 3}, 
           {'松紧': 0, '拉链': 1, '系带': 2}, 
           {'松紧带': 0, '拉链': 1, '套筒': 2, '套脚': 2, '一脚蹬': 2, '系带': 3, '魔术贴': 4, '搭扣': 5}, 
           {'高帮': 0, '中帮': 0, '低帮': 1}]

def get_coef(n):
    dq = deque([[]])
    dq_next = deque([])
    for i in range(n):
        while dq:
            tmp = dq.popleft()
            dq_next.append(tmp + [1])
            dq_next.append(tmp + [0])
        dq = dq_next
        dq_next = deque([])
    return dq

def evaluate(dataset, model):
    model.eval()
    scores_, labels_ = [], []
    acc_match_pos, acc_match_dual_neg, acc_match_neg,  tp_attr2, pos_num2  = 0, 0, 0, [], []
    for input in dataset:
        input = [f.cuda() for f in input]
        acc_match_pos_batch, acc_match_dual_neg_batch, acc_match_neg_batch, \
            tp_attr2_batch, pos_num2_batch = model.getMetric(input) 
        acc_match_pos += sum(acc_match_pos_batch>0.5)
        acc_match_dual_neg += sum(acc_match_dual_neg_batch<0.5)
        acc_match_neg += sum(acc_match_neg_batch<0.5)
        
        scores_.extend(list(acc_match_pos_batch)+list(acc_match_dual_neg_batch))
        labels_.extend([1]*len(acc_match_pos_batch) + [0] * len(acc_match_dual_neg_batch))
        
        tp_attr2.append(tp_attr2_batch)
        pos_num2.append(pos_num2_batch) 
            
    auc_score = roc_auc_score(labels_, scores_)
    
    acc_match_pos_precision = acc_match_pos/5000
    acc_match_dual_neg_precision = acc_match_dual_neg / 5000
    acc_match_neg_precision = acc_match_neg/1412
    tp_attr2, pos_num2 = np.array(tp_attr2), np.array(pos_num2)
    tp_attr2_cate = np.sum(tp_attr2, axis=0)
    pos_num2_cate = np.sum(pos_num2, axis=0)
    
    # precision = all_attr_precision*0.5 + acc_match_precision * 0.5
    # precision = (acc_match_pos_precision + acc_match_neg_precision) /2 * 0.5 + sum(tp_attr2_cate)/sum(pos_num2_cate) * 0.5
    precision = auc_score * 0.5 + sum(tp_attr2_cate)/sum(pos_num2_cate) * 0.5
    print(f"图文匹配pos acc: {acc_match_pos_precision}")
    print(f"图文匹配pos对应的neg acc: {acc_match_dual_neg_precision}")
    print(f"图文匹配neg acc: {acc_match_neg_precision}")
    print(f"图文匹配auc: {auc_score}")
   
    print(f"attr acc: {tp_attr2_cate/pos_num2_cate}")
    print(f"attr 总的acc: {sum(tp_attr2_cate)/sum(pos_num2_cate)}")
    print(f"加权acc: {precision}")
    print(f"各标签数: {pos_num2_cate}")
    print("============================================")
    return precision

class EMA():
    def __init__(self, model, decay=0.95):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# # 初始化
# ema = EMA(model, 0.999)
# ema.register()

# # 训练过程中，更新完参数后，同步update shadow weights
# def train():
#     optimizer.step()
#     ema.update()

# # eval前，apply shadow weights；eval之后，恢复原来模型的参数
# def evaluate():
#     ema.apply_shadow()
#     # evaluate
#     ema.restore()


class FocalLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, pred, target):  
        # pred: bsz, C
        p1 = torch.sigmoid(-pred)
        p2 = torch.sigmoid(pred)
        loss_pos = -1 * p1 * torch.log(p2+1e-12) * target.float()
        loss_neg = -1 * p2 * torch.log(p1+1e-12) * ( 1 - target).float()
        
        loss = torch.mean(torch.sum(loss_pos + loss_neg, dim=-1))
        
        return loss
        
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.5, emb_name='bert.embeddings.word_embeddings.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='bert.embeddings.word_embeddings.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {} 