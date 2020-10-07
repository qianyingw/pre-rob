#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 17:01:02 2020

@author: qwang
"""

import os
import json
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# from transformers import DistilBertConfig # DistilBertForSequenceClassification  # DistilBertTokenizer
# from transformers import BertForSequenceClassification  # BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from arg_parser import get_args
import utils
from helper import RobDataset, BatchTokenizer
from model import DistilClsLinear, DistilClsLSTM, DistilClsConv, BertClsLinear
from train import train_fn, valid_fn


#%% random seed
args = get_args()
random.seed(args.seed)
#np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True     
torch.backends.cudnn.benchmark = False   # will be slower  

#%% Data loader
train_set = RobDataset(info_dir = args.info_dir, pkl_dir = args.pkl_dir, 
                       rob_item = args.rob_item, rob_sent = args.rob_sent, 
                       max_n_sent = args.max_n_sent,
                       group='train')

valid_set = RobDataset(info_dir = args.info_dir, pkl_dir = args.pkl_dir, 
                       rob_item = args.rob_item, rob_sent = args.rob_sent, 
                       max_n_sent = args.max_n_sent,
                       group='valid')

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=BatchTokenizer())
valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=BatchTokenizer())

#%% Model & Optimizer & Scheduler & Criterion
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if args.model == 'distil_linear':
    model = DistilClsLinear.from_pretrained('distilbert-base-uncased', return_dict=True)
if args.model == 'distil_lstm':    
    model = DistilClsLSTM.from_pretrained('distilbert-base-uncased', return_dict=True)
if args.model == 'distil_conv':    
    model = DistilClsConv.from_pretrained('distilbert-base-uncased', return_dict=True)
if args.model == 'bert':
    model = BertClsLinear.from_pretrained('bert-base-uncased', return_dict=True)
    
model.to(device)
optimizer = AdamW(model.parameters(), lr=args.lr)

# Slanted triangular Learning rate scheduler
total_steps = len(train_loader) * args.num_epochs // args.accum_step
warm_steps = int(total_steps * args.warm_frac)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_steps, num_training_steps=total_steps)


# loss function (w/o weight balancing)
if args.wgt_bal == True and torch.cuda.device_count() == 0:
    loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor(train_set.cls_weight()))
elif args.wgt_bal == True and torch.cuda.device_count() > 0:
    loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor(train_set.cls_weight()).cuda())
else:
    loss_fn = nn.CrossEntropyLoss()

#%% Train the model
if os.path.exists(args.exp_dir) == False:
    os.makedirs(args.exp_dir)   
       
# Create args and output dictionary (for json output)
output_dict = {'args': vars(args), 'prfs': {}}

# For early stopping
n_worse = 0
# min_valid_loss = float('inf')
max_valid_f1 = -float('inf')

for epoch in range(args.num_epochs):   
    train_scores = train_fn(model, train_loader, optimizer, scheduler, loss_fn, utils.metrics_fn, args.clip, args.accum_step, args.threshold, device)
    valid_scores = valid_fn(model, valid_loader, loss_fn, utils.metrics_fn, args.threshold, device)

    # Update output dictionary
    output_dict['prfs'][str('train_'+str(epoch+1))] = train_scores
    output_dict['prfs'][str('valid_'+str(epoch+1))] = valid_scores
       
    # Save scores
    # if valid_scores['loss'] < min_valid_loss:
    #     min_valid_loss = valid_scores['loss']    
    if valid_scores['f1'] > max_valid_f1:
        max_valid_f1 = valid_scores['f1'] 
        
    is_best = valid_scores['f1'] > max_valid_f1
    if is_best == True:       
        utils.save_dict_to_json(valid_scores, os.path.join(args.exp_dir, 'best_val_scores.json'))
    
    # Save model
    if args.save_model == True:
        utils.save_checkpoint({'epoch': epoch+1,
                               'state_dict': model.state_dict(),
                               'optim_Dict': optimizer.state_dict()},
                               is_best = is_best, checkdir = args.exp_dir)
    
    print("\n\nEpoch {}/{}...".format(epoch+1, args.num_epochs))                       
    print('\n[Train] loss: {0:.3f} | acc: {1:.2f}% | f1: {2:.2f}% | rec: {3:.2f}% | prec: {4:.2f}% | spec: {5:.2f}%'.format(
            train_scores['loss'], train_scores['accuracy']*100, train_scores['f1']*100, train_scores['recall']*100, train_scores['precision']*100, train_scores['specificity']*100))
    print('[Valid] loss: {0:.3f} | acc: {1:.2f}% | f1: {2:.2f}% | rec: {3:.2f}% | prec: {4:.2f}% | spec: {5:.2f}%\n'.format(
        valid_scores['loss'], valid_scores['accuracy']*100, valid_scores['f1']*100, valid_scores['recall']*100, valid_scores['precision']*100, valid_scores['specificity']*100))
    
    # Early stopping             
    # if valid_scores['loss']-min_valid_loss > 0: # args.stop_c1) and (max_valid_f1-valid_scores['f1'] > args.stop_c2):
    #     n_worse += 1
    # if n_worse == 5: # args.stop_p:
    #     print("Early stopping")
    #     break
        
# Write performance and args to json
prfs_name = os.path.basename(args.exp_dir)+'_prfs.json'
prfs_path = os.path.join(args.exp_dir, prfs_name)
with open(prfs_path, 'w') as fout:
    json.dump(output_dict, fout, indent=4)

#%% plot
utils.plot_prfs(prfs_path) 