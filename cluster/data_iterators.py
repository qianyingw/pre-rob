#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 15:53:39 2019

@author: qwang
"""


import os
import json
import random

from torchtext import data
import torchtext.vocab as vocab
import torch





#%%
class DataIterators(object):
    
    def __init__(self, args_dict):
        
        """
        Params:
            arg_json_path: path of args json file
            data_json_path: path of data json file
            embed_path: path of pre-trained word embeddings
      
        The dataset json is read and splitted into three jsons: "train.json", "val.json", "test.json".
        
        """
        
        self.args_dict = args_dict
        
#        # Load parameters 
#        assert os.path.isfile(args_json_path), "No args json file exist."
#        with open(args_json_path) as fin:
#            self.args_dict = json.load(fp=fin)
            
        # Create data field
        self.ID = data.Field()    
        self.TEXT = data.Field()    
        self.LABEL = data.LabelField()
        
    
    
    def split_and_save(self):
        """
        Read tokenized dataset in json format
        Shuffle and split data
        Write to separate json files
              
        """  
        data_json_path = self.args_dict['data_json_path']        
        if os.path.exists(data_json_path) == False:
            raise("Data doesn't exist: {}".format(os.path.basename(data_json_path)))
        
        dat = []
        with open(data_json_path, 'r') as fin:
            for line in fin:
                dat.append(json.loads(line))        
        
        random.seed(self.args_dict['seed'])
        random.shuffle(dat)
        
        train_size = int(len(dat) * self.args_dict['train_ratio'])
        val_size = int(len(dat) * self.args_dict['val_ratio'])
        
        train_list = dat[:train_size]
        val_list = dat[train_size : (train_size + val_size)]
        test_list = dat[(train_size + val_size):]
         
        
        data_dir = os.path.dirname(self.args_dict['data_json_path'])
        
        with open(os.path.join(data_dir, 'train.json'), 'w') as fout:
            for dic in train_list:     
                fout.write(json.dumps(dic) + '\n')
            
        with open(os.path.join(data_dir, 'val.json'), 'w') as fout:
            for dic in val_list:     
                fout.write(json.dumps(dic) + '\n')
        
        with open(os.path.join(data_dir, 'test.json'), 'w') as fout:
            for dic in test_list:     
                fout.write(json.dumps(dic) + '\n')   
          
        
        
    def create_data(self):
        """
        Create train/valid/test data
        
        """
        rob_item = self.args_dict['rob_name']
        
        
        if rob_item == 'random': rob_item = 'RandomizationTreatmentControl'
        if rob_item == 'blinded': rob_item = 'BlindedOutcomeAssessment'
        if rob_item == 'ssz': rob_item = 'SampleSizeCalculation'
        
        fields = {'goldID': ('id', self.ID), 
                  rob_item: ('label', self.LABEL),
                  'textTokens': ('text', self.TEXT)}

        train_data, valid_data, test_data = data.TabularDataset.splits(path = os.path.dirname(self.args_dict['data_json_path']),
                                                                       train = 'train.json',
                                                                       validation = 'val.json',
                                                                       test = 'test.json',
                                                                       format = 'json',
                                                                       fields = fields)
        return train_data, valid_data, test_data
        
    
    def load_embedding(self):
        
        embed_path = self.args_dict['embed_path']        
        custom_embedding = vocab.Vectors(name = os.path.basename(embed_path), 
                                         cache = os.path.dirname(embed_path))
        return custom_embedding
    
    
    def build_vocabulary(self, train_data, valid_data, test_data):
        self.ID.build_vocab(train_data, valid_data, test_data)
        self.LABEL.build_vocab(train_data)
        self.TEXT.build_vocab(train_data,
                              max_size = self.args_dict['max_vocab_size'],
                              min_freq = self.args_dict['min_occur_freq'],
                              vectors = self.load_embedding(),
                              unk_init = torch.Tensor.normal_)
    
    
    def create_iterators(self, train_data, valid_data, test_data):
        
        self.build_vocabulary(train_data, valid_data, test_data)
        
        ## CUDA
        if torch.cuda.is_available():  # checks whether a cuda gpu is available and whether the gpu flag is True
            device = torch.cuda.current_device()
        else:
            device = torch.device('cpu')
        
        
        train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
            (train_data, valid_data, test_data),
            sort = False,
            shuffle = True,
            batch_size = self.args_dict['batch_size'],
            device = device
        )
        
        return train_iterator, valid_iterator, test_iterator
        
#%% Instance   
#args_dict = {'seed': 1234,
#             'batch_size': 64,
#             'num_epochs': 2,
#             'train_ratio': 0.8,
#             'val_ratio': 0.1,
#             'max_vocab_size': 100,
#             'min_occur_freq': 10,
#             'embed_dim': 200,
#             'num_filters': 20,
#             'filter_sizes': '2,3,4',
#             'dropout': 0.5,
#             'exp_path': '/home/qwang/rob/src/cluster/exps',
#             'exp_name': 'cnn1',
#             'rob_name': 'blinded',
#             'use_gpu': False,
#             'gpu_id': 'None',
#             'args_json_path': None,
#             'embed_path': '/media/mynewdrive/rob/wordvec/wikipedia-pubmed-and-PMC-w2v.txt',
##             'data_json_path': '/media/mynewdrive/rob/data/stroke/rob_stroke_fulltokens.json',
#             'data_json_path': '/media/mynewdrive/rob/data/rob_gold_tokens.json',
#             'use_cuda': False}
#
#helper = DataIterators(args_dict = args_dict)
## Generate train/valid/test.json
#helper.split_and_save()
#train_data, valid_data, test_data = helper.create_data()   
#train_iterator, valid_iterator, test_iterator = helper.create_iterators(train_data, valid_data, test_data)

#print(helper.LABEL.vocab.stoi)  # {0: 0, 1: 1} ~= {'No': 0, 'Yes': 1}
## helper.ID.vocab.stoi  # {'<unk>': 0, '<pad>': 1, 'psy1': 2, 'psy10': 3, ..., 'psy999': 2405}
#helper.ID.vocab.stoi['<pad>']  # 1
#helper.ID.vocab.stoi['psy1']  # 2297
#helper.ID.vocab.stoi['stroke999']  # 7783
#
#helper.TEXT.vocab.itos[:5]  # ['<unk>', '<pad>', ',', '.', 'the']
#
#len(helper.TEXT.vocab)  # max_vocab_size + 2
#len(helper.LABEL.vocab)  # 2
#len(helper.ID.vocab)  # 7782 + 2
#
#helper.TEXT.pad_token  # '<pad>'
#helper.TEXT.unk_token  # '<unk>'
#helper.TEXT.vocab.stoi[helper.TEXT.pad_token]  # 1
#helper.TEXT.vocab.stoi[helper.TEXT.unk_token]  # 0

