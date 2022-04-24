from cProfile import label
from os import truncate
from unittest.util import _MAX_LENGTH
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader 
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
import json
import numpy as np
import copy
import random

# 松紧、拉链、系带 会有重复, 但是分属不同的品类: 裤门襟(裤子)  闭合方式(鞋子)

# 各个key attr的 precision: [0.80526919 0.96488294 0.68948247 0.88888889 0.96153846 0.99212598
#  0.81927711 0.81092437 0.99636364 0.85148515 0.95486111 0.91693291]
# 总的attr precision: 0.8684450524395805
# 加权precision: 0.7157225262197902
# 各个key attr标签数: [873. 598. 599. 738.  26. 254.  83. 238. 275. 101. 288. 313.]
# 版型 裙长  裤型 裤门襟

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

replace_val = { '高领': ['半高领', '立领'], '半高领': ['高领', '立领'], '立领': ['半高领', '高领'],
               '翻领': ['衬衫领', 'POLO领', '方领', '娃娃领', '荷叶领'], 
               '衬衫领': ['翻领', 'POLO领', '方领', '娃娃领', '荷叶领'], 
               'POLO领': ['翻领', '衬衫领', '方领', '娃娃领', '荷叶领'], 
               '方领': ['翻领', '衬衫领', 'POLO领', '娃娃领', '荷叶领'], 
               '娃娃领': ['翻领', '衬衫领', 'POLO领', '方领', '荷叶领'], 
               '荷叶领': ['翻领', '衬衫领', 'POLO领', '方领', '娃娃领'],
               '连帽': ['可脱卸帽'], '可脱卸帽': ['连帽'], 
               '短袖': ['五分袖'], '五分袖': ['短袖'], '九分袖': ['长袖'], '长袖': ['九分袖'],
               '超短款': ['短款', '常规款'], '短款': ['超短款', '常规款'], '常规款': ['超短款', '短款'], 
               '长款': ['超长款'], '超长款': ['长款'],
               '修身型': ['标准型'], '标准型': ['修身型'], 
               '短裙': ['超短裙'], '超短裙': ['短裙'], '中裙': ['中长裙'], '中长裙': ['中裙'],
               'O型裤': ['锥形裤', '哈伦裤', '灯笼裤'], '锥形裤': ['O型裤', '哈伦裤', '灯笼裤'], '哈伦裤': ['O型裤', '锥形裤', '灯笼裤'], '灯笼裤': ['O型裤', '锥形裤', '哈伦裤'],
               '铅笔裤': ['直筒裤', '小脚裤'], '直筒裤': ['铅笔裤', '小脚裤'], '小脚裤': ['铅笔裤', '直筒裤'],
               '喇叭裤': ['微喇裤'], '微喇裤': ['喇叭裤'], 
               '九分裤': ['长裤'], '长裤': ['九分裤'],
               '套筒': ['套脚', '一脚蹬'], '套脚': ['套筒', '一脚蹬'], '一脚蹬': ['套筒', '套脚'],
               '高帮': ['中帮'], '中帮': ['高帮']         
}

label_nums = [14, 4, 3, 2, 3, 2, 4, 7, 4, 3, 6, 2]

class MyDataSet(Dataset):
    def __init__(self, path, tokenizer=None, mode='fine') -> None:
        self.mode = mode
        self.tokenizer = tokenizer
        self.label_nums = [14, 4, 3, 2, 3, 2, 4, 7, 4, 3, 6, 2]
        self.imgs, self.texts, self.label_match, self.label_attr , self.tasks_mask, self.task_names = self.read(path)
        # label_attr: sample_number, 12
        # tasks_mask: sample_number, 12
    
    def read(self, path):
        images, texts, label_match, label_attr, tasks_mask, task_names = [], [], [], [], [], []  # label_atrr:  samples*num_tasks
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = json.loads(line.strip())
                attrs = line['key_attr']  # coarse 中存在key_attr为空的  1029个 属性不匹配  10000个图文不匹配
                # if not attrs:
                #     continue
                images.append(line['feature'])
                texts.append(line['title'])
                img_text_match = line['match'].get('图文', 0)
                label_match.append(img_text_match)
                task_names.append(attrs)
                tasks_mask_ = [0] * 12
                label_attr_ = [0] * 12
                # if not label_match:
                #     continue
                # 图文匹配的数据才有属性匹配，图文不匹配 属性是否匹配未知
               
                if True:
                    for key in attrs:
                        # if self.mode == 'coarse' and key == '衣长':
                        #     continue
                        tasks_mask_[tasksMap[key]] = 1
                        label_attr_[tasksMap[key]] = valsMap[tasksMap[key]][attrs[key]]
                    
                label_attr.append(label_attr_)
                tasks_mask.append(tasks_mask_)
        return images, texts, label_match, label_attr, tasks_mask, task_names
                    
                
    def __len__(self):
        return len(self.label_match)
    
    def __getitem__(self, idx):
        # 正样本增强
        title = self.texts[idx]
        match = self.label_match[idx]
        pos_title_mask, neg_title_mask = 1, 1
        # for val in self.task_names[idx].values():
        #     if val in replace_val and np.random.rand() < 0.3:
        #         title = title.replace(val, np.random.choice(replace_val[val]))
        
        text_encode = self.tokenizer(title, padding=True, truncation=True, max_length=32, return_attention_mask=True)
        text_ids, text_mask = text_encode['input_ids'], text_encode['attention_mask']
        # generate negative text： 只用作图文匹配的负样本
        neg_title = title
        neg_tasks_mask = copy.deepcopy(self.tasks_mask[idx])
        if len(self.task_names[idx]) > 0 and match: 
            select_task = np.random.choice(list(self.task_names[idx].keys()))
            neg_tasks_mask = copy.deepcopy(self.tasks_mask[idx])
            neg_tasks_mask[tasksMap[select_task]] = 0
            for key in self.task_names[idx].keys():
                if key != select_task:
                    if np.random.rand() < 0.3:   # 再取0.2的概率随机替换
                        val = np.random.choice(list(valsMap[tasksMap[key]].keys()))
                        if val != self.task_names[idx][key]:
                            neg_title = neg_title.replace(self.task_names[idx][key], val)
                            neg_tasks_mask[tasksMap[key]] = 0
            
            while True:
                select_attr_val = np.random.choice(list(valsMap[tasksMap[select_task]].keys()))
                if valsMap[tasksMap[select_task]][select_attr_val] != valsMap[tasksMap[select_task]][self.task_names[idx][select_task]]:
                    neg_title = neg_title.replace(self.task_names[idx][select_task], select_attr_val)
                    break
        
        if len(self.task_names) < 1 and match:
            neg_title_mask = 0
        
        if not match:
            pos_title_mask = 0
            
        # if match and np.random.rand() < 0.4:
        #     neg_title_mask = 0
        
        neg_text_encode =  self.tokenizer(neg_title, padding=True, truncation=True, max_length=32, return_attention_mask=True)
        neg_text_ids, neg_text_mask = neg_text_encode['input_ids'], neg_text_encode['attention_mask']
        
        #  属性上的负样本 
        pos_attr_title =  title
        pos_tasks_mask = copy.deepcopy(self.tasks_mask[idx])
        for key in self.task_names[idx].keys():
            if np.random.rand() > 0.5: 
                # if self.task_names[idx][key] in replace_val and np.random.rand() < 0.3:
                #     pos_attr_title = pos_attr_title.replace(self.task_names[idx][key], np.random.choice(replace_val[self.task_names[idx][key]]))
                pass
            else:
                while True:
                    val = np.random.choice(list(valsMap[tasksMap[key]].keys()))  # 随机替换
                    if valsMap[tasksMap[key]][val] != valsMap[tasksMap[key]][self.task_names[idx][key]]:
                        pos_tasks_mask[tasksMap[key]] = 0
                        pos_attr_title = pos_attr_title.replace(self.task_names[idx][key], val)
                        break
        
        pos_attr_text_encode =  self.tokenizer(pos_attr_title, padding=True, truncation=True, max_length=32, return_attention_mask=True)
        pos_attr_text_ids, pos_attr_text_mask = pos_attr_text_encode['input_ids'], pos_attr_text_encode['attention_mask']

        return torch.tensor(self.imgs[idx]), torch.tensor(text_ids), torch.tensor(text_mask),  torch.tensor(self.label_attr[idx]), torch.tensor(self.tasks_mask[idx]), \
                torch.tensor(neg_text_ids), torch.tensor(neg_text_mask), torch.tensor(neg_tasks_mask), \
                torch.tensor(pos_attr_text_ids), torch.tensor(pos_attr_text_mask), torch.tensor(pos_tasks_mask), torch.tensor(pos_title_mask), \
                torch.tensor(neg_title_mask)
    
    @classmethod
    def collate_fn(cls, x):
        imgs = torch.stack([sample[0] for sample in x], dim=0)
        text_ids = pad_sequence([sample[1] for sample in x], batch_first=True)
        text_mask = pad_sequence([sample[2] for sample in x], batch_first=True)
        label_attr = torch.stack([sample[3] for sample in x], dim=0)
        tasks_mask = torch.stack([sample[4] for sample in x], dim=0)
        neg_text_ids = pad_sequence([sample[5] for sample in x], batch_first=True)
        neg_text_mask = pad_sequence([sample[6] for sample in x], batch_first=True)
        neg_tasks_mask = torch.stack([sample[7] for sample in x], dim=0)
        pos_attr_text_ids = pad_sequence([sample[8] for sample in x], batch_first=True)
        pos_attr_text_mask = pad_sequence([sample[9] for sample in x], batch_first=True)
        pos_tasks_mask = torch.stack([sample[10] for sample in x], dim=0)
        pos_title_mask = torch.stack([sample[11] for sample in x], dim=0)
        neg_title_mask = torch.stack([sample[12] for sample in x], dim=0)
        return imgs, text_ids, text_mask, label_attr, tasks_mask, neg_text_ids, neg_text_mask, neg_tasks_mask, \
            pos_attr_text_ids, pos_attr_text_mask, pos_tasks_mask, pos_title_mask, neg_title_mask
    
    
class TestDataSet(Dataset):
    def __init__(self, path, tokenizer=None) -> None:
        self.tokenizer = tokenizer
        self.label_nums = [14, 4, 3, 2, 3, 2, 4, 7, 4, 3, 6, 2]
        self.names, self.imgs, self.texts, self.label_match, self.label_attr, self.tasks_mask = self.read(path)
    
    def read(self, path):
        names, images, texts, label_match, label_attr, tasks_mask =[], [], [], [], [], [] # label_atrr:  samples*num_tasks
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = json.loads(line.strip())
                names.append(line['img_name'])
                images.append(line['feature'])
                texts.append(line['title'])
                attrs = line['query']
                tasks_mask_ = [0] * 12
                label_attr_ = [0] * 12
                for key in attrs:
                    if key == '图文':
                        pass
                    else:
                        tasks_mask_[tasksMap[key]] = 1
                        for val in valsMap[tasksMap[key]]:
                            if val in line['title']:
                                label_attr_[tasksMap[key]] = valsMap[tasksMap[key]][val]
                                break
                            
                label_attr.append(label_attr_)
                tasks_mask.append(tasks_mask_)
        return names, images, texts, label_match, label_attr, tasks_mask
                    
                
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        text_encode = self.tokenizer(self.texts[idx], padding=True, truncation=True, max_length=32, return_attention_mask=True)
        text_ids, text_mask = text_encode['input_ids'], text_encode['attention_mask']
        return torch.tensor(self.imgs[idx]), torch.tensor(text_ids), torch.tensor(text_mask), torch.tensor(self.label_attr[idx]), torch.tensor(self.tasks_mask[idx])
    
    @classmethod
    def collate_fn(cls, x):
        imgs = torch.stack([sample[0] for sample in x], dim=0)
        text_ids = pad_sequence([sample[1] for sample in x], batch_first=True)
        text_mask = pad_sequence([sample[2] for sample in x], batch_first=True)
        label_attr = torch.stack([sample[3] for sample in x], dim=0)
        tasks_mask = torch.stack([sample[4] for sample in x], dim=0)
        return imgs, text_ids, text_mask, label_attr, tasks_mask
    

class PretrainDataSet(Dataset):
    def __init__(self, path, tokenizer=None, mode='fine') -> None:
        self.mode = mode
        self.tokenizer = tokenizer
        self.label_nums = [14, 4, 3, 2, 3, 2, 4, 7, 4, 3, 6, 2]
        self.imgs, self.texts, self.label_match, self.label_attr , self.tasks_mask, self.task_names = self.read(path)
        # label_attr: sample_number, 12
        # tasks_mask: sample_number, 12
    
    def read(self, path):
        images, texts, label_match, label_attr, tasks_mask, task_names = [], [], [], [], [], []  # label_atrr:  samples*num_tasks
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = json.loads(line.strip())
                attrs = line['key_attr']  # coarse 中存在key_attr为空的  1029个 属性不匹配  10000个图文不匹配
                if not attrs:
                    continue
                images.append(line['feature'])
                texts.append(line['title'])
                img_text_match = line['match'].get('图文', 0)
                label_match.append(img_text_match)
                task_names.append(attrs)
                tasks_mask_ = [0] * 12
                label_attr_ = [0] * 12
                # if not label_match:
                #     continue
                # 图文匹配的数据才有属性匹配，图文不匹配 属性是否匹配未知
                if img_text_match:
                    for key in attrs:
                        # if self.mode == 'coarse' and key == '衣长':
                        #     continue
                        tasks_mask_[tasksMap[key]] = 1
                        label_attr_[tasksMap[key]] = valsMap[tasksMap[key]][attrs[key]]
                    
                label_attr.append(label_attr_)
                tasks_mask.append(tasks_mask_)
        return images, texts, label_match, label_attr, tasks_mask, task_names
                    
                
    def __len__(self):
        return len(self.label_match)
    
    def __getitem__(self, idx):
        # 正样本增强
        title = self.texts[idx]
        select_task = np.random.choice(list(self.task_names[idx].keys()))
        for key in self.task_names[idx].keys():
            if np.random.rand() > 0.3: 
                # if self.task_names[idx][key] in replace_val and np.random.rand() < 0.3:
                #     pos_attr_title = pos_attr_title.replace(self.task_names[idx][key], np.random.choice(replace_val[self.task_names[idx][key]]))
                pass
            elif key != select_task:
                while True:
                    val = np.random.choice(list(valsMap[tasksMap[key]].keys()))  # 随机替换
                    if valsMap[tasksMap[key]][val] != valsMap[tasksMap[key]][self.task_names[idx][key]]:
                        title = title.replace(self.task_names[idx][key], val)
                        break

        text_encode = self.tokenizer(title, padding=True, truncation=True, max_length=40, return_attention_mask=True)
        text_ids, text_mask = text_encode['input_ids'], text_encode['attention_mask']

        label_ids, label_mask = [-100] * len(text_ids), [0] * len(text_mask)
        mask_id = self.tokenizer._convert_token_to_id_with_added_voc(self.tokenizer.mask_token)
        val = self.task_names[idx][select_task]
        start = title.index(val)
        ids_start, ids_end = text_encode.char_to_token(start), text_encode.char_to_token(start+len(val)-1)
        if not ids_end:
            ids_end = 38   # 此时 title进行了截断，ids_end = max_length-1
        label_ids[ids_start: ids_end + 1] = text_ids[ids_start: ids_end + 1]
        label_mask[ids_start: ids_end + 1] = [1] * (ids_end - ids_start + 1)
        if np.random.rand() < 0.8:
            text_ids[ids_start: ids_end + 1] = [mask_id] * (ids_end - ids_start + 1)  
        else:
            val_sub = np.random.choice(list(valsMap[tasksMap[select_task]].keys()))
            val_ids = self.tokenizer(val_sub, add_special_tokens=False)['input_ids'][:(ids_end-ids_start+1)]
            val_ids = val_ids + (ids_end - ids_start + 1 - len(val_ids)) * [mask_id]
            text_ids[ids_start: ids_end + 1] = val_ids
        
        return torch.tensor(text_ids), torch.tensor(text_mask), torch.tensor(label_ids), torch.tensor(self.imgs[idx])

    
    @classmethod
    def collate_fn(cls, x):
        text_ids = pad_sequence([sample[0] for sample in x], batch_first=True)
        text_mask = pad_sequence([sample[1] for sample in x], batch_first=True)
        
        label_ids = pad_sequence([sample[2] for sample in x], batch_first=True)
        # label_mask = pad_sequence([sample[3] for sample in x], batch_first=True)
        
        imgs = torch.stack([sample[3] for sample in x], dim=0)
        
        return text_ids, text_mask, label_ids, imgs
    