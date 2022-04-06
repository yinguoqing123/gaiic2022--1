import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoConfig, AutoTokenizer, BertModel, BertTokenizer
from torch.utils.data import DataLoader, SequentialSampler, ConcatDataset
import numpy as np
from dataset import MyDataSet
from model import MyModel
from utils import evaluate

bert_name = 'M-CLIP/M-BERT-Distil-40'
bert_name = 'sentence-transformers/clip-ViT-B-32-multilingual-v1'

state_dict = torch.load("../pretrained_model/clip-ViT-B-32-multilingual-v1.bin")
tokenizer = AutoTokenizer.from_pretrained(bert_name)
bert = AutoModel.from_pretrained(bert_name, state_dict=state_dict)
# tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
# bert = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext')


model = MyModel(bert)
model.load_state_dict(torch.load("../model/model_best.pt"))
model = model.cuda()

path_train = '../data/train/train_fine.txt.00'
path_coarse_train = '../data/train/train_coarse_trans.txt'
path_test = '../data/train/train_fine.txt.01'
trainset = MyDataSet(path_train, tokenizer=tokenizer)
traincoarseset = MyDataSet(path_coarse_train, tokenizer=tokenizer)
trainsetunion = ConcatDataset([trainset, traincoarseset])
testset = MyDataSet(path_test, tokenizer=tokenizer)
testsample = SequentialSampler(testset)

trainload = DataLoader(trainsetunion, batch_size=128, shuffle=True, collate_fn=trainset.collate_fn)
testload = DataLoader(testset, batch_size=128, sampler=testsample, collate_fn=testset.collate_fn)

bert_parameters = list(model.bert.parameters())
other_parameters = []
for name, param in model.named_parameters():
    if 'bert' not in name and param.requires_grad:
        other_parameters.append(param)

p = [{'params': bert_parameters, 'lr': 3e-5}, {'params': other_parameters, 'lr': 1e-4}]   
optimizer = torch.optim.Adam(p)
lrscheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.8)

best_p = 0.0
for epoch in range(20):
    step = 0
    running_loss = 0
    for input in trainload:
        model.train()
        step += 1
        input = [f.cuda() for f in input]
        loss = model(input)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if step % 100 == 99:
            print(f"Epoch {epoch+1}, step {step+1} : {running_loss}")
            running_loss = 0
        
        if step % 300 == 299:
            p = evaluate(testload, model)
            if p > best_p:
                p = best_p
                torch.save(model.state_dict(), f'../model/model_best.pt')
                
    lrscheduler.step()
                
