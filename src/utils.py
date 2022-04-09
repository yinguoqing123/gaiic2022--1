
import torch
import numpy as np

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

def evaluate(dataset, model):
    model.eval()
    acc_match, attr_tp, attr_posnum = 0, [], []
    neg_attr_tp = 0
    for input in dataset:
        input = [f.cuda() for f in input]
        acc_match_batch , attr_tp_batch, attr_posnum_batch = model.getMetric(input) 
        acc_match += acc_match_batch
        attr_tp.append(attr_tp_batch)
        attr_posnum.append(attr_posnum_batch)
        
    attr_tp, attr_posnum = np.array(attr_tp), np.array(attr_posnum)
    attr_tp_cate = np.sum(attr_tp, axis=0)
    attr_posnum_cate = np.sum(attr_posnum, axis=0)
    all_attr_precision = sum(attr_tp_cate) / sum(attr_posnum_cate)
    
    acc_match_precision = acc_match/2000
    
    precision = all_attr_precision*0.5 + acc_match_precision * 0.5
    
    print(f"图文匹配batch内top1 acc: {acc_match_precision}")
    print(f"各个key attr的 precision: {attr_tp_cate/attr_posnum_cate}")
    print(f"总的attr precision: {all_attr_precision}")
    print(f"加权precision: {precision}")
    print(f"各个key attr标签数: {attr_posnum_cate}")
    print("============================================")
    return precision