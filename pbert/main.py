#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 17:45:31 2020

@author: qwang
"""


import os
import json
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import BertConfig, BertTokenizer, AdamW


import utils
from arg_parser import get_args
from data_loader import RobDataset, PadDoc
from model import BertClsPLinear, BertClsPConv, BertClsPLSTM
from train import train_fn, valid_fn, test_fn


#%% Setting
# Get arguments from command line
args = get_args()

# random seed
random.seed(args.seed)
#np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True     
torch.backends.cudnn.benchmark = False   # This makes things slower  

# device
if torch.cuda.device_count() > 1:
    device = torch.cuda.current_device()
elif torch.cuda.device_count() == 1:
    device = torch.device("cuda")
else:
    device = torch.device('cpu')     


#%% Tokenizer & Config & Model   
# Default: biobert
if args.pre_wgts == "biobert":
    args.pre_wgts = "dmis-lab/biobert-v1.1"
if args.pre_wgts == "pubmed-full":
    args.pre_wgts = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
if args.pre_wgts == "pubmed-abs":
    args.pre_wgts = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

# Tokenizer
tokenizer = BertTokenizer.from_pretrained(args.pre_wgts)  

# Config
config = BertConfig.from_pretrained(args.pre_wgts)  
config.output_hidden_states = True
config.num_labels = 2
config.unfreeze = args.unfreeze
config.pool_layers = args.pool_layers
config.pool_method = args.pool_method


if args.model == "bert_pool_lstm":
    model = BertClsPLSTM.from_pretrained(args.pre_wgts, config=config)
if args.model == "bert_pool_conv":
    config.num_filters = args.num_filters
    sizes = args.filter_sizes.split(',')
    config.filter_sizes = [int(s) for s in sizes]
    model = BertClsPConv.from_pretrained(args.pre_wgts, config=config)
if args.model == "bert_pool_linear":
    config.pool_method_chunks = args.pool_method_chunks
    model = BertClsPLinear.from_pretrained(args.pre_wgts, config=config)



#%% Create dataset and data loader  
train_set = RobDataset(info_dir = args.info_dir, pkl_dir = args.pkl_dir, rob_item = args.rob_item, 
                       max_chunk_len = args.max_chunk_len, max_n_chunk = args.max_n_chunk,
                       group = 'train', tokenizer = tokenizer)

doc = train_set[0][0]
valid_set = RobDataset(info_dir = args.info_dir, pkl_dir = args.pkl_dir, rob_item = args.rob_item, 
                       max_chunk_len = args.max_chunk_len, max_n_chunk = args.max_n_chunk,
                       group = 'valid', tokenizer = tokenizer)

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=PadDoc())
valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=PadDoc())

if args.save_model:
    test_set = RobDataset(info_dir = args.info_dir, pkl_dir = args.pkl_dir, rob_item = args.rob_item, 
                          max_chunk_len = args.max_chunk_len, max_n_chunk = args.max_n_chunk,
                          group = 'test', tokenizer = tokenizer)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=PadDoc())
        
    
#%% 
# Optimizer
optimizer = AdamW(model.parameters(), lr = args.lr, eps = 1e-8)

# Slanted triangular Learning rate scheduler
from transformers import get_linear_schedule_with_warmup
total_steps = len(train_loader) * args.num_epochs // args.accum_step
warm_steps = int(total_steps * args.warm_frac)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_steps, num_training_steps=total_steps)


# Loss function (weight balancing)
if args.wgt_bal == True and torch.cuda.device_count() == 0:
    loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor(train_set.cls_weight()))
elif args.wgt_bal == True and torch.cuda.device_count() > 0:
    loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor(train_set.cls_weight()).cuda())
else:
    loss_fn = nn.CrossEntropyLoss()

# Sent to device
if torch.cuda.device_count() > 1:  # multiple GPUs
    model = nn.DataParallel(module=model)
model = model.to(device)
loss_fn = loss_fn.to(device)    

#%% Train the model
if os.path.exists(args.exp_dir) == False:
    os.makedirs(args.exp_dir)     

if args.restore_file is not None:
    restore_path = os.path.join(args.exp_dir, args.restore_file + '.pth.tar')
    utils.load_checkpoint(restore_path, model, optimizer)
       
# Create args and output dictionary (for json output)
output_dict = {'args': vars(args), 'prfs': {}}
max_valid_f1 = -float('inf')

for epoch in range(args.num_epochs):   
    train_scores = train_fn(model, train_loader, optimizer, scheduler, loss_fn, utils.metrics_fn, device, args.clip, args.accum_step, args.threshold)
    valid_scores = valid_fn(model, valid_loader, loss_fn, utils.metrics_fn, device, args.threshold)        

    # Update output dictionary
    output_dict['prfs'][str('train_'+str(epoch+1))] = train_scores
    output_dict['prfs'][str('valid_'+str(epoch+1))] = valid_scores
    
    
    is_best = valid_scores['f1'] > max_valid_f1
    if is_best == True:
        max_valid_f1 = valid_scores['f1']
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
    print('[Val] loss: {0:.3f} | acc: {1:.2f}% | f1: {2:.2f}% | rec: {3:.2f}% | prec: {4:.2f}% | spec: {5:.2f}%\n'.format(
        valid_scores['loss'], valid_scores['accuracy']*100, valid_scores['f1']*100, valid_scores['recall']*100, valid_scores['precision']*100, valid_scores['specificity']*100))
    
# Write performance and args to json
prfs_name = os.path.basename(args.exp_dir)+'_prfs.json'
prfs_path = os.path.join(args.exp_dir, prfs_name)
with open(prfs_path, 'w') as fout:
    json.dump(output_dict, fout, indent=4)
    
#%% Test
if args.save_model:
    test_scores = test_fn(model, test_loader, loss_fn, utils.metrics_fn, args, device, restore_file = 'best')   