import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoConfig, AutoTokenizer
from torch.utils.data import DataLoader, SequentialSampler
import numpy as np
from dataset import MyDataSet, TestDataSet
from model import MyModel
from utils import evaluate
import json

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

label_nums = [14, 4, 3, 2, 3, 2, 4, 7, 4, 3, 6, 2]

bert_name = 'M-CLIP/M-BERT-Distil-40'
bert_name = 'sentence-transformers/clip-ViT-B-32-multilingual-v1'

state_dict = torch.load("../pretrained_model/clip-ViT-B-32-multilingual-v1.bin")
tokenizer = AutoTokenizer.from_pretrained(bert_name)
bert = AutoModel.from_pretrained(bert_name, state_dict=state_dict)


model = MyModel(bert)
model.load_state_dict(torch.load("../model/model_best.pt"))
model = model.cuda()

path = '../data/preliminary_testA.txt'
dataset = TestDataSet(path, tokenizer=tokenizer)
datasample = SequentialSampler(dataset)

dataloader = DataLoader(dataset, batch_size=64, sampler=datasample, collate_fn=dataset.collate_fn)

model.eval()
match_label, attr_match = [], []
tasks_array = np.array(['领型', '袖长', '衣长', '版型', '裙长', '穿着方式', '类别', '裤型', '裤长', '裤门襟', 
        '闭合方式', '鞋帮高度'])


for input in dataloader:
    label_attr, mask = input[-2:]  # mask: bsz, num_task
    label_attr = label_attr.cpu().numpy()
    mask = mask.cpu().numpy()
    input = input[:-2]
    input = [f.cuda() for f in input]
    imgtextscore, attrscore = model.getSubmit(input)  # attrscore: bsz, num_task
    imgtextscore = np.where(imgtextscore>0.4, 1, 0)
    match_label.extend(imgtextscore)
    for i in range(len(attrscore)):
        task_nm = tasks_array[mask[i]==1]
        task_val = attrscore[i][mask[i]==1]
        flag = (label_attr[i][mask[i]==1] == task_val).astype(int)
        tmp = list(zip(task_nm, flag))
        attr_match.append(tmp)
        
img_name = dataset.names

with open("../data/submit.json", "w", encoding='utf-8') as f:
    for i in range(len(img_name)):
        d = {'img_name': img_name[i]}
        attr = {}
        attr['图文'] = 1 if int(match_label[i]) > 0.65 else 0
        for query, val in attr_match[i]:
            attr[query] = int(val)
        d['match'] = attr
        d = json.dumps(d, ensure_ascii=False)
        f.write(d+'\n')
        
    