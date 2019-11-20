#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Fri Sep 27 16:37:58 2019
@author: qwang
"""
import os
os.getcwd()
os.chdir('/home/qwang/rob')

import json
import random
import itertools
from tqdm import tqdm


import torch
from torchtext import data
import torchtext.vocab as vocab
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


#%% Setting
SEED = 1234
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

MAX_VOCAB_SIZE = 100
MIN_OCCUR_FREQ = 10
N_FILTERS = 20
FILTER_SIZES = [2,3,4]
DROPOUT = 0.5
EMBEDDING_DIM = 200
#OUTPUT_DIM = 1

BATCH_SIZE = 32
N_EPOCHS = 1


random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#%% Split json file
#filename = 'data/psycho/rob_psycho_fulltokens.json'
#psy = []
#with open(filename, 'r') as fin:
#    for line in fin:
#        psy.append(json.loads(line))
#
#random.shuffle(psy)
#train_size = int(len(psy) * TRAIN_RATIO)
#val_size = int(len(psy) * VAL_RATIO)
#
#train_list = psy[:train_size]
#val_list = psy[train_size : (train_size + val_size)]
#test_list = psy[(train_size + val_size):]
#
#
#with open('data/psycho/train.json', 'w') as fout:
#    for dic in train_list:     
#        fout.write(json.dumps(dic) + '\n')
#    
#with open('data/psycho/val.json', 'w') as fout:
#    for dic in val_list:     
#        fout.write(json.dumps(dic) + '\n')
#
#with open('data/psycho/test.json', 'w') as fout:
#    for dic in test_list:     
#        fout.write(json.dumps(dic) + '\n')   
    
# Question: should I shuffle samples by class?


#%% Create data
ID = data.Field()    
TEXT = data.Field()    
LABEL = data.LabelField()


#    For json data, we must create a dictionary where: 
#        the key matches the key of the json object
#        the value is a tuple where:
#            the first element becomes the batch object's attribute name
#            the second element is the name of the Field


fields = {'goldID': ('id', ID), 
          'RandomizationTreatmentControl': ('label', LABEL),
          'textTokens': ('text', TEXT)}


train_data, valid_data, test_data = data.TabularDataset.splits(path = 'data/psycho',
                                                               train = 'train.json',
                                                               validation = 'val.json',
                                                               test = 'test.json',
                                                               format = 'json',
                                                               fields = fields)

#%% Load custom embedding
custom_embedding = vocab.Vectors(name = 'wikipedia-pubmed-and-PMC-w2v.txt',
                                 cache = 'wordvec')


#%% Build vocabulary
ID.build_vocab(train_data, valid_data, test_data)
LABEL.build_vocab(train_data)
TEXT.build_vocab(train_data,
                 max_size = MAX_VOCAB_SIZE,
                 min_freq = MIN_OCCUR_FREQ,
                 vectors = custom_embedding,
                 unk_init = torch.Tensor.normal_)

# print(LABEL.vocab.stoi)  # {0: 0, 1: 1} ~= {'No': 0, 'Yes': 1}
# print(ID.vocab.stoi)  # {'<unk>': 0, '<pad>': 1, 'psy1': 2, 'psy10': 3, ..., 'psy999': 2405}

# print(TEXT.vocab.itos[:5])  # ['<unk>', '<pad>', ',', '.', 'the']
# dict(itertools.islice(TEXT.vocab.stoi.items(), 6))  # {'<unk>': 0, '<pad>': 1, ',': 2, '.': 3, 'the': 4, 'of': 5}

# psy = pd.read_csv("data/psycho/rob_psycho_info.txt", sep='\t', engine="python", encoding="utf-8", index_col = 0)  
# psy.loc[psy.goldID == ID.vocab.itos[1051], 'RandomizationTreatmentControl']


#%% Create iterators
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
            (train_data, valid_data, test_data),
            sort = False,
            shuffle = True,
            batch_size = BATCH_SIZE,
            device = device
        )

#for batch in valid_iterator:
#    print(batch)

#    [torchtext.data.batch.Batch of size 64]
#            [.id]:[torch.LongTensor of size 1x64]
#            [.label]:[torch.LongTensor of size 64]
#            [.text]:[torch.LongTensor of size 18900x64]
#    
#    [torchtext.data.batch.Batch of size 64]
#            [.id]:[torch.LongTensor of size 1x64]
#            [.label]:[torch.LongTensor of size 64]
#            [.text]:[torch.LongTensor of size 19526x64]
#    
#    [torchtext.data.batch.Batch of size 64]
#            [.id]:[torch.LongTensor of size 1x64]
#            [.label]:[torch.LongTensor of size 64]
#            [.text]:[torch.LongTensor of size 22188x64]
#    
#    [torchtext.data.batch.Batch of size 48]
#            [.id]:[torch.LongTensor of size 1x48]
#            [.label]:[torch.LongTensor of size 48]
#            [.text]:[torch.LongTensor of size 17317x48]


#for batch in valid_iterator:
#    print(batch.id)

#for batch in test_iterator:
#    print(batch.id)
    

#%% Model
class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(in_channels = 1,
                                              out_channels = n_filters,
                                              kernel_size = (fsize, embedding_dim)) for fsize in filter_sizes
                                   ])
                
        self.fc = nn.Linear(n_filters * len(filter_sizes), output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # text: [seq_len, batch_size]
        embed = self.embedding(text)  # [seq_len, batch_size, embedding_dim]
        embed = embed.permute(1,0,2)  # [batch_size, seq_len, embedding_dim]
        embed = embed.unsqueeze(1)  # [batch_size, 1, seq_len, embedding_dim]
        
        conved = [F.relu(conv(embed)) for conv in self.convs]  # [batch_size, n_filters, (seq_len-fsize+1), 1]
        conved = [conv.squeeze(3) for conv in conved]  # [batch_size, n_filters, (seq_len-fsize+1)]
        pooled = [F.max_pool1d(conv, conv.shape[2]) for conv in conved]  # [batch_size, n_filters, 1]
        pooled = [pool.squeeze(2) for pool in pooled]  # [batch_size, n_filters]
        
        cat = torch.cat(pooled, dim=1)  # [batch_size, n_filters * len(filter_sizes)]
        dp = self.dropout(cat)
        out = self.fc(dp)
        
        return out
        


#%% Metrics
def metrics(preds, y):
    y_preds = preds.argmax(dim=1, keepdim=False)  # [batch_size, output_dim]  --> [batch_size]
        
    ones = torch.ones_like(y_preds)
    zeros = torch.zeros_like(y_preds)
    
    pos = torch.eq(y_preds, y).sum().item()
    tp = (torch.eq(y_preds, ones) & torch.eq(y, ones)).sum().item()
    tn = (torch.eq(y_preds, zeros) & torch.eq(y, zeros)).sum().item()
    fp = (torch.eq(y_preds, ones) & torch.eq(y, zeros)).sum().item()
    fn = (torch.eq(y_preds, zeros) & torch.eq(y, ones)).sum().item()
    
    assert pos == tp + tn
    
    acc = pos / y.shape[0]  # torch.FloatTensor([y.shape[0]])
    f1 = 2*tp / (tp + tn + fp + fn) if (tp + tn + fp + fn != 0) else float('nan')
    rec = tp / (tp + fn) if (tp + fn != 0) else float('nan')
    prec = tp / (tp + fp) if (tp + fp != 0) else float('nan')
    spec = tn / (tn + fp) if (tn + fp != 0) else float('nan')
       
    return acc, f1, rec, prec, spec


def confusion(preds, y):
    y_preds = preds.argmax(dim=1, keepdim=False)  # [batch_size, output_dim]  --> [batch_size]
        
    ones = torch.ones_like(y_preds)
    zeros = torch.zeros_like(y_preds)

    tp = (torch.eq(y_preds, ones) & torch.eq(y, ones)).sum().item()
    tn = (torch.eq(y_preds, zeros) & torch.eq(y, zeros)).sum().item()
    fp = (torch.eq(y_preds, ones) & torch.eq(y, zeros)).sum().item()
    fn = (torch.eq(y_preds, zeros) & torch.eq(y, ones)).sum().item() 
    
    return tp, tn, fp, fn

#%% Training
INPUT_DIM = len(TEXT.vocab)
OUTPUT_DIM = len(LABEL.vocab)   

PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

model = CNN(vocab_size = INPUT_DIM,
            embedding_dim = EMBEDDING_DIM, 
            n_filters = N_FILTERS, 
            filter_sizes = FILTER_SIZES, 
            output_dim = OUTPUT_DIM, 
            dropout = DROPOUT, 
            pad_idx = PAD_IDX)

print(model)

# Load pre-trained embedding
pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)
# Zero the initial weights of the unknown and padding tokens
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)



optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

model = model.to(device)
criterion = criterion.to(device)

def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    epoch_f1 = 0
    epoch_rec = 0
    epoch_prec = 0
    epoch_spec = 0
    
    model.train()
    
    with tqdm(total=len(iterator)) as progress_bar:
        for batch in iterator:
            optimizer.zero_grad()
            preds = model(batch.text)
            loss = criterion(preds, batch.label)
            
            acc, f1, rec, prec, spec = metrics(preds, batch.label)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc
            epoch_f1 += f1
            epoch_rec += rec
            epoch_prec += prec
            epoch_spec += spec
            
            progress_bar.update(1)
        
    loss_out = epoch_loss / len(iterator)
    acc_out = epoch_acc / len(iterator)
    f1_out = epoch_rec / len(iterator)
    rec_out = epoch_f1 / len(iterator)
    prec_out = epoch_prec / len(iterator)
    spec_out = epoch_spec / len(iterator)
    
    return loss_out, acc_out, f1_out, rec_out, prec_out, spec_out


def validate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    epoch_f1 = 0
    epoch_rec = 0
    epoch_prec = 0
    epoch_spec = 0
   
    model.eval()
    
    with torch.no_grad():
        with tqdm(total=len(iterator)) as progress_bar:
            for batch in iterator:
                preds = model(batch.text)
                
                loss = criterion(preds, batch.label)
                acc, f1, rec, prec, spec = metrics(preds, batch.label)
                epoch_loss += loss.item()
                epoch_acc += acc
                epoch_f1 += f1
                epoch_rec += rec
                epoch_prec += prec
                epoch_spec += spec
                
                progress_bar.update(1)
        
    loss_out = epoch_loss / len(iterator)
    acc_out = epoch_acc / len(iterator)
    f1_out = epoch_rec / len(iterator)
    rec_out = epoch_f1 / len(iterator)
    prec_out = epoch_prec / len(iterator)
    spec_out = epoch_spec / len(iterator)
    
    return loss_out, acc_out, f1_out, rec_out, prec_out, spec_out

#%% Main
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    train_loss, train_acc, train_f1, train_rec, train_prec, train_spec = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc, valid_f1, valid_rec, valid_prec, valid_spec = validate(model, valid_iterator, criterion)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'random-model.pt')
    
    print('[Epoch {0} / Train]: loss: {1:.3f} | acc: {2:.2f}% | f1: {3:.2f}% | recall: {4:.2f}% | precision: {5:.2f}% | specificity: {6:.2f}%'.format(
        epoch+1, train_loss, train_acc*100, train_f1*100, train_rec*100, train_prec*100, train_spec*100))
    print('[Epoch {0} / Val]: loss: {1:.3f} | acc: {2:.2f}% | f1: {3:.2f}% | recall: {4:.2f}% | precision: {5:.2f}% | specificity: {6:.2f}%'.format(
        epoch+1, valid_loss, valid_acc*100, valid_f1*100, valid_rec*100, valid_prec*100, valid_spec*100))
    
    
    
model.load_state_dict(torch.load('random-model.pt'))
test_loss, test_acc, test_f1, test_rec, test_prec, test_spec = validate(model, test_iterator, criterion)

print('[Test]: test_loss: {0:.3f} | acc: {1:.2f}% | f1: {2:.2f}% | recall: {3:.2f}% | precision: {4:.2f}% | specificity: {5:.2f}%'.format(
        test_loss, test_acc*100, test_f1*100, test_rec*100, test_prec*100, test_spec*100))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



    
    
    
    