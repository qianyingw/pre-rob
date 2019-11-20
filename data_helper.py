#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 11:16:20 2019
@author: qwang
"""

import os
import json
import random

from torchtext import data
import torchtext.vocab as vocab
import torch

import utils


#%%
class DataHelper(object):
    
    def __init__(self, data_dir, data_name, params_dir, embed_dir, embed_name, params):
        """
        Params:
            json_dir: directory containing json data
            json_name: file name of the single json data file
            params: (Params) hyperparameters of the training process. 
                    This function modifies params and appends dataset_params (such as hidden_dim etc.) to params.
                    
        One json file is for dataset, another one is for parameters which is called "params.json".
        The dataset json is read and splitted into three jsons: "train.json", "val.json", "test.json".
        
        """
        
        self.data_dir = data_dir
        self.data_name = data_name
        self.params_dir = params_dir
        self.embed_dir = embed_dir
        self.embed_name = embed_name
        
        # Load parameters
        params_path = os.path.join(params_dir, 'params.json')
        assert os.path.isfile(params_path), "'params.json' doesn't exist"
        self.params = utils.Params(params_path)
        
        # Create data field
        self.ID = data.Field()    
        self.TEXT = data.Field()    
        self.LABEL = data.LabelField()
        
        # Add dataset parameters to param
        params.update(params_path)
    
    
    def split_and_save(self):
        """
        Read tokenized dataset in json format
        Shuffle and split data
        Write to separate json files
              
        """             
        json_file = os.path.join(self.data_dir, self.data_name)
        if os.path.exists(json_file) == False:
            raise("File doesn't exist: {}".format(self.data_name))
        
        dat = []
        with open(json_file, 'r') as fin:
            for line in fin:
                dat.append(json.loads(line))
        
        
        random.seed(self.params.seed)
        random.shuffle(dat)
        
        train_size = int(len(dat) * self.params.train_ratio)
        val_size = int(len(dat) * self.params.val_ratio)
        
        train_list = dat[:train_size]
        val_list = dat[train_size : (train_size + val_size)]
        test_list = dat[(train_size + val_size):]
                
        with open(os.path.join(self.data_dir, 'train.json'), 'w') as fout:
            for dic in train_list:     
                fout.write(json.dumps(dic) + '\n')
            
        with open(os.path.join(self.data_dir, 'val.json'), 'w') as fout:
            for dic in val_list:     
                fout.write(json.dumps(dic) + '\n')
        
        with open(os.path.join(self.data_dir, 'test.json'), 'w') as fout:
            for dic in test_list:     
                fout.write(json.dumps(dic) + '\n')   
          
        
        
    def create_data(self, rob_item):
        """
        Create train/valid/test data
        Params: 
            json_dir: directory containing train.json, val.json and test.json
            rob_item: (string) risk of bias item name
        
        """
        fields = {'goldID': ('id', self.ID), 
                  rob_item: ('label', self.LABEL),
                  'textTokens': ('text', self.TEXT)}

        train_data, valid_data, test_data = data.TabularDataset.splits(path = self.data_dir,
                                                                       train = 'train.json',
                                                                       validation = 'val.json',
                                                                       test = 'test.json',
                                                                       format = 'json',
                                                                       fields = fields)
        return train_data, valid_data, test_data
        
    
    def load_embedding(self):
        custom_embedding = vocab.Vectors(name = self.embed_name, cache = self.embed_dir)
        return custom_embedding
    
    
    def build_vocabulary(self, train_data, valid_data, test_data):
        self.ID.build_vocab(train_data, valid_data, test_data)
        self.LABEL.build_vocab(train_data)
        self.TEXT.build_vocab(train_data,
                              max_size = self.params.max_vocab_size,
                              min_freq = self.params.min_occur_freq,
                              vectors = self.load_embedding(),
                              unk_init = torch.Tensor.normal_)
    
    
    def create_iterators(self, train_data, valid_data, test_data):
        self.build_vocabulary(train_data, valid_data, test_data)
        train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
            (train_data, valid_data, test_data),
            sort = False,
            shuffle = True,
            batch_size = self.params.batch_size,
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        return train_iterator, valid_iterator, test_iterator
        
#%% Instance   
#params = utils.Params(json_path = 'src/model/params.json')
#print(params.n_epochs)
#
#
#params.cuda = True
#params.n_epochs = 4
#print(params.n_epochs)
#params.save(json_path = 'src/model/params.json')
#print(params.n_epochs)
#
#helper = DataHelper(data_dir = 'data/psycho', data_name = 'rob_psycho_fulltokens.json',
#                    params_dir = 'src/model',
#                    embed_dir = 'wordvec', embed_name = 'wikipedia-pubmed-and-PMC-w2v.txt', 
#                    params = params)
#
#train_data, valid_data, test_data = helper.create_data(rob_item = 'RandomizationTreatmentControl')
#train_iterator, valid_iterator, test_iterator = helper.create_iterators(train_data, valid_data, test_data)
#
#print(helper.LABEL.vocab.stoi)  # {0: 0, 1: 1} ~= {'No': 0, 'Yes': 1}
## helper.ID.vocab.stoi  # {'<unk>': 0, '<pad>': 1, 'psy1': 2, 'psy10': 3, ..., 'psy999': 2405}
#helper.ID.vocab.stoi['<pad>']  # 1
#helper.ID.vocab.stoi['psy1']  # 2
#helper.ID.vocab.stoi['psy999']  # 2405
#
#helper.TEXT.vocab.itos[:5]  # ['<unk>', '<pad>', ',', '.', 'the']
#
#len(helper.TEXT.vocab)  
#len(helper.LABEL.vocab)
#len(helper.ID.vocab)  # 2406
#
#helper.TEXT.pad_token  # '<pad>'
#helper.TEXT.unk_token  # '<unk>'
#helper.TEXT.vocab.stoi[helper.TEXT.pad_token]  # 1
#helper.TEXT.vocab.stoi[helper.TEXT.unk_token]  # 0
