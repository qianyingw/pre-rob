#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 18:24:36 2019

@author: qwang
"""

import os
import random
import logging
import json

import torch
import torch.nn as nn
import torch.optim as optim

# os.chdir('/home/qwang/rob/src/cluster')

import utils
from utils import metrics
from arg_parser import get_args
from data_iterators import DataIterators

from model import ConvNet, RecurNet, AttnNet
from model_han import HAN
from model_transformer import TransformerNet
from train import train_evaluate, test #, plot_performance


#%% Get arguments from command line
args = get_args()


#%% Set random seed and device
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True     
torch.backends.cudnn.benchmark = False   # This makes things slower  


# device
if torch.cuda.device_count() > 1:
    device = torch.cuda.current_device()
    print('Use {} GPUs'.format(torch.cuda.device_count()), device)
elif torch.cuda.device_count() == 1:
    device = torch.device("cuda")
    print('Use 1 GPU', device)
else:
    device = torch.device('cpu')     

   
#%% Set logger
log_dir = os.path.join(args.exp_path, args.exp_name)
if os.path.exists(log_dir) == False:
    os.makedirs(log_dir)       
#utils.set_logger(os.path.join(log_dir, 'train.log'))


#%% Save args to json
args_dict = vars(args)
with open(os.path.join(log_dir, 'args.json'), 'w') as fout:
    json.dump(args_dict, fout, indent=4)
        
        
#%% Load data and create iterators
logging.info("Loading the datasets...")
helper = DataIterators(args_dict = args_dict)

# Create train/valid/test.json
json_dir = os.path.dirname(args.data_json_path)
helper.split_and_save() 

# Create data
train_data, valid_data, test_data = helper.create_data()   

# Create iterators
train_iterator, valid_iterator, test_iterator = helper.create_iterators(train_data, valid_data, test_data)
logging.info("Done.")

#%% Define the model
input_dim = len(helper.TEXT.vocab)  # max_vocab_size + 2
output_dim = len(helper.LABEL.vocab)  # 2

unk_idx = helper.TEXT.vocab.stoi[helper.TEXT.unk_token]  # 0
pad_idx = helper.TEXT.vocab.stoi[helper.TEXT.pad_token]  # 1


if args.net_type == 'cnn':
    sizes = args_dict['filter_sizes'].split(',')
    sizes = [int(s) for s in sizes]
    model = ConvNet(vocab_size = input_dim,
                    embedding_dim = args.embed_dim, 
                    n_filters = args.num_filters, 
                    filter_sizes = sizes, 
                    output_dim = output_dim, 
                    dropout = args.dropout, 
                    pad_idx = pad_idx)

if args.net_type == 'rnn':
    model = RecurNet(vocab_size = input_dim, 
                     embedding_dim = args.embed_dim, 
                     rnn_hidden_dim = args.rnn_hidden_dim, 
                     rnn_num_layers = args.rnn_num_layers, 
                     output_dim = output_dim, 
                     bidirection = args.bidirection, 
                     rnn_cell_type = args.rnn_cell_type,
                     dropout = args.dropout, 
                     pad_idx = pad_idx)
    
if args.net_type == 'attn':
    model = AttnNet(vocab_size = input_dim, 
                    embedding_dim = args.embed_dim, 
                    rnn_hidden_dim = args.rnn_hidden_dim, 
                    rnn_num_layers = args.rnn_num_layers, 
                    output_dim = output_dim, 
                    bidirection = args.bidirection, 
                    rnn_cell_type = args.rnn_cell_type, 
                    dropout = args.dropout, 
                    pad_idx = pad_idx)

if args.net_type == 'han':
    model = HAN(vocab_size = input_dim,
                embedding_dim = args.embed_dim,
                word_hidden_dim = args.word_hidden_dim,
                word_num_layers = args.word_num_layers,
                pad_idx = pad_idx,            
                sent_hidden_dim = args.sent_hidden_dim,
                sent_num_layers = args.sent_num_layers,
                output_dim = output_dim)

    
if args.net_type == 'transformer':
    model = TransformerNet(vocab_size = input_dim, 
                           embedding_dim = args.embed_dim, 
                           num_heads = args.num_heads, 
                           num_encoder_layers = args.num_encoder_layers, 
                           output_dim = output_dim, 
                           pad_idx = pad_idx)
print(model)

#%% Load pre-trained embedding
pretrained_embeddings = helper.TEXT.vocab.vectors

if args.net_type == 'han':
    model.word_attn.embedding.weight.data.copy_(pretrained_embeddings) 
    model.word_attn.embedding.weight.data[unk_idx] = torch.zeros(args.embed_dim)  # Zero the initial weights for <unk> tokens
    model.word_attn.embedding.weight.data[pad_idx] = torch.zeros(args.embed_dim)  # Zero the initial weights for <pad> tokens
else:
    model.embedding.weight.data.copy_(pretrained_embeddings)
    model.embedding.weight.data[unk_idx] = torch.zeros(args.embed_dim)  # Zero the initial weights for <unk> tokens
    model.embedding.weight.data[pad_idx] = torch.zeros(args.embed_dim)  # Zero the initial weights for <pad> tokens

del pretrained_embeddings

#%% Define the optimizer, loss function and metrics
optimizer = optim.Adam(model.parameters())
metrics_fn = metrics

# Weight balancing
if args.weight_balance == True:
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(helper.cls_weight).cuda())
else:
    criterion = nn.CrossEntropyLoss()
 
    
if torch.cuda.device_count() > 1:  # multiple GPUs
    model = nn.DataParallel(module=model)
model = model.to(device)
criterion = criterion.to(device)

#%% Train the model
logging.info("\nStart training for {} epoch(s)...".format(args.num_epochs)) 
train_evaluate(model, train_iterator, valid_iterator, criterion, optimizer, metrics_fn, args, log_dir)

#%% Test
if args.save_model:
    logging.info("\nStart testing...")
    test_scores = test(model, test_iterator, criterion, metrics_fn, log_dir, restore_file = 'best')
    
    # Add test performance to '_prfs.json'
    prfs_path = os.path.join(log_dir, args.exp_name+'_prfs.json')
    with open(prfs_path) as fin:
        output_dict = json.load(fp=fin)
    output_dict['prfs']['test'] = test_scores    
    with open(prfs_path, 'w') as fout:
        json.dump(output_dict, fout, indent=4)

#%% Performance plot
# plot_performance(train_df, valid_df, png_dir = log_dir)
