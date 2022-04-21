import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoConfig, BertModel, AutoTokenizer,  BertTokenizer, BertTokenizerFast
from torch.utils.data import DataLoader, SequentialSampler, ConcatDataset
import numpy as np
from dataset import MyDataSet, PretrainDataSet
from model import MyModel
from utils import evaluate, EMA
from visualbert import DistilBertModel, DistilBertForMaskedLM
from lebert import BertModel, BertForMaskedLM

# bert_name = 'sentence-transformers/clip-ViT-B-32-multilingual-v1'
# config = AutoConfig.from_pretrained(bert_name)
# state_dict = torch.load("../pretrained_model/clip-ViT-B-32-multilingual-v1.bin")
# tokenizer = AutoTokenizer.from_pretrained(bert_name)
# bert = DistilBertModel.from_pretrained(bert_name, state_dict=state_dict)
# model = DistilBertForMaskedLM(config, bert)
# model.distilbert.load_state_dict(state_dict, strict=False)

bert_name = 'hfl/rbt3'
state_dict = torch.load("../pretrained_model/rbt3_mlm.bin")
tokenizer = BertTokenizerFast.from_pretrained(bert_name)
model = BertForMaskedLM.from_pretrained(bert_name, state_dict=state_dict)
# # bert = BertModel.from_pretrained(bert_name, state_dict=state_dict)


model = model.cuda()
# ema = EMA(model)
# # model.load_state_dict(torch.load("../model/model_best_pretrained.pt"))
# ema.register()


path_train = '../data/train/train_fine.txt.00'
path_coarse_train = '../data/train/train_coarse_trans.txt'
path_test = '../data/train/train_fine.txt.01'

trainset = PretrainDataSet(path_train, tokenizer=tokenizer)
traincoarseset = PretrainDataSet(path_coarse_train, tokenizer=tokenizer, mode='coarse')

testset = PretrainDataSet(path_test, tokenizer=tokenizer)
testsample = SequentialSampler(testset)

trainsetunion = ConcatDataset([trainset, traincoarseset])
trainload = DataLoader(trainsetunion, batch_size=128, shuffle=True, collate_fn=trainset.collate_fn, num_workers=8)
testload = DataLoader(testset, batch_size=128, sampler=testsample, collate_fn=testset.collate_fn, num_workers=8)

no_decay = ["bias", "LayerNorm.weight", 'layer_norm', 'layernorm']
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.0001,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=5e-5) 

# optimizer = torch.optim.Adam(p)
#lrscheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.7, patience=2)

@torch.no_grad()
def evaluate(data, model):
    loss_ = 0
    tp, num = 0, 0
    for input in data:
        model.eval()
        input = [f.cuda() for f in input]
        text_ids, text_mask, label_ids, imgs = input
        text_mask = torch.cat([text_mask, torch.ones(text_mask.shape[0], 1, device='cuda', dtype=torch.int64)], dim=-1)
        label_ids = torch.cat([label_ids, torch.full((label_ids.shape[0], 1), -100, device='cuda', dtype=torch.int64)], dim=-1)
        loss = model(text_ids, text_mask, labels=label_ids, visual_embeds=imgs) 
        loss_ += loss[0].item()
        pred_label = loss[1].argmax(dim=-1)
        tp += torch.sum((label_ids==pred_label).float()*(pred_label>0).float()).item() 
        num += torch.sum(label_ids>0).item()
    return loss_, tp/num

min_loss, char_acc = evaluate(testload, model)

for epoch in range(20):
    step = 0
    running_loss = 0
    for input in trainload:
        model.train()
        step += 1
        input = [f.cuda() for f in input]
        text_ids, text_mask, label_ids, imgs = input
        text_mask = torch.cat([text_mask, torch.ones(text_mask.shape[0], 1, device='cuda', dtype=torch.int64)], dim=-1)
        label_ids = torch.cat([label_ids, torch.full((label_ids.shape[0], 1), -100, device='cuda', dtype=torch.int64)], dim=-1)
        loss = model(text_ids, text_mask, labels=label_ids, visual_embeds=imgs)[0]
        # optimizer.zero_grad()
        loss.backward()
        if step % 4 == 0:
            optimizer.step()
            optimizer.zero_grad()
            
        running_loss += loss.item() 
        if step % 100 == 0:
            print(f"Epoch {epoch+1}, step {step+1} : {running_loss}")
            running_loss = 0
        
        if step % 200 == 0:
            
            # if epoch >= 4 and ema_first:
            #     ema.register()
            #     ema_first = False
                
            cur_loss, char_acc = evaluate(testload, model)
            print(f"val set loss: {cur_loss}, char acc: {char_acc}")
            if cur_loss < min_loss:
                min_loss = cur_loss
                torch.save(model.state_dict(), f'../model/model_best_pretrained.pt')
                # if not ema_first:
                #     ema.update()
                #     print(" ----  ema更新权重 -----")
                #     ema.apply_shadow()
                #     p = evaluate(testload, model)
                #     if p > best_p:
                #         best_p = p
                #         torch.save(model.state_dict(), f'../model/model_best.pt')
                #     ema.restore()
                
            scheduler.step(cur_loss)
            
    cur_loss, char_acc = evaluate(testload, model)
    if cur_loss < min_loss:
        min_loss = cur_loss
        print(f"val set loss: {cur_loss}, char acc: {char_acc}")
        torch.save(model.state_dict(), f'../model/model_best_pretrained.pt')
    