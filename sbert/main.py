#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 17:01:02 2020

@author: qwang
"""


import torch
from torch.utils.data import DataLoader

from transformers import BertTokenizer, BertForSequenceClassification

from arg_parser import get_args
from helper import RobDataset, BatchTokenizer


args = get_args()

#%%
train_set = RobDataset(info_dir = args.info_dir, pkl_dir = args.pkl_dir, 
                       rob_item = args.rob_item, rob_sent = args.rob_sent, 
                       max_n_sent = args.max_n_sent,
                       group='train')

valid_set = RobDataset(info_dir = args.info_dir, pkl_dir = args.pkl_dir, 
                       rob_item = args.rob_item, rob_sent = args.rob_sent, 
                       max_n_sent = args.max_n_sent,
                       group='valid')



# temp = train_set[0][0]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=BatchTokenizer())


model = BertForSequenceClassification.from_pretrained('bert-base-uncased', return_dict=True)
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
outputs = model(**inputs, labels=labels)
loss = outputs.loss
logits = outputs.logits
