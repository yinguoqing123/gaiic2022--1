
import torch
import numpy as np

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