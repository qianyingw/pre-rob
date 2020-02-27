#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 15:53:39 2019

@author: qwang
"""


import os
import json
import random
import math

from torchtext import data
import torchtext.vocab as vocab
import torch

#%%
class DataIterators(object):
    
    def __init__(self, args_dict):
        
        """
        Params:
            arg_dict: ...
        The dataset json is read and splitted into three jsons: "train.json", "val.json", "test.json".
        
        """
        
        self.args_dict = args_dict
            
        # Create data field
        self.ID = data.Field()
        self.LABEL = data.LabelField()
        
        if self.args_dict['net_type'] == 'han':           
            max_sent_len = self.args_dict['max_sent_len'] if self.args_dict['max_sent_len'] != 0 else None
            max_doc_len = self.args_dict['max_doc_len'] if self.args_dict['max_doc_len'] != 0 else None
            # nested sentence tokens
            nest_field = data.Field(pad_token='<pad>', fix_length = max_sent_len)  # fix num of words in each sent (fix max_sent_len)
            self.TEXT = data.NestedField(nest_field, fix_length = max_doc_len)  # fix num of sents (fix max_doc_len)
        else:
            self.TEXT = data.Field()   # word tokens 
        
        
        # Modify rob name
        self.rob_item = self.args_dict['rob_name']       
        if self.rob_item == 'random': self.rob_item = 'RandomizationTreatmentControl'
        if self.rob_item == 'blind': self.rob_item = 'BlindedOutcomeAssessment'
        if self.rob_item == 'size': self.rob_item = 'SampleSizeCalculation'
        if self.rob_item == 'exclusion': self.rob_item = 'AnimalExclusions'     
        if self.rob_item == 'conceal': self.rob_item = 'AllocationConcealment'
        if self.rob_item == 'welfare': self.rob_item = 'AnimalWelfareRegulations'
        if self.rob_item == 'conflict': self.rob_item = 'ConflictsOfInterest'
            
 
    
    def split_and_save(self):
        """
        Read tokenized dataset in json format
        Shuffle and split data
        Write to separate json files
              
        """  
        data_json_path = self.args_dict['data_json_path']        

        
        dat = []
        try: 
            with open(data_json_path, 'r') as fin:
                for line in fin:
                    dat.append(json.loads(line))     
        except:
            print("Data doesn't exist: {}".format(os.path.basename(data_json_path)))
           
        # Remove records with NA annotations
        dat = [g for g in dat if math.isnan(g[self.rob_item]) == False]    
        print('Overal data size: {}'.format(len(dat)))
        
        # Calculate weight for balancing data
        num_pos = len([g for g in dat if g[self.rob_item] == 1])
        num_neg = len([g for g in dat if g[self.rob_item] == 0])
        self.cls_weight = [1/num_neg, 1/num_pos]
    
        # Cut sequence
        if self.args_dict['max_token_len'] != 0:
            for d in dat:
                if len(d['wordTokens']) > self.args_dict['max_token_len']:
                    d['wordTokens'] = d['wordTokens'][:self.args_dict['max_token_len']]
            
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
        
        if self.args_dict['net_type'] == 'han':  	
            fields = {'goldID': ('id', self.ID), 
        			  self.rob_item: ('label', self.LABEL), # 'label': ('label', self.LABEL) for rob data
        			  'sentTokens': ('text', self.TEXT)}		
        else:
            fields = {'goldID': ('id', self.ID), 
                       self.rob_item: ('label', self.LABEL), 
                       'wordTokens': ('text', self.TEXT)}
            

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
        
        # !custom_embedding.stoi['cat'])
        # !custom_embedding.vectors[6])

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
        if torch.cuda.is_available(): 
            device = torch.device('cuda') # torch.cuda.current_device() 
        else:
            device = torch.device('cpu')
        
        
        train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
            (train_data, valid_data, test_data),
            sort = False,
            shuffle = True,
            batch_size = self.args_dict['batch_size'],
            device = device
        )
        
#        train_iterator, valid_iterator = data.BucketIterator.splits(
#            (train_data, valid_data),
#            sort = False,
#            shuffle = True,
#            batch_size = self.args_dict['batch_size'],
#            device = device
#        )
#        
#        
#        test_iterator = data.Iterator(test_data, 
#                                      sort = False,
#                                      sort_within_batch = False, 
#                                      shuffle = False,
#                                      batch_size = self.args_dict['batch_size'], 
#                                      device = device, 
#                                      repeat=False)
        
        return train_iterator, valid_iterator, test_iterator
        
#%% Instance   
#args_dict = {'seed': 1234,
#             'batch_size': 32,
#             'num_epochs': 2,
#             'train_ratio': 0.8,
#             'val_ratio': 0.1,
#             'max_vocab_size': 5000,
#             'min_occur_freq': 10,
#             'max_token_len': 0,
#             'embed_dim': 200,
#             'dropout': 0.5,            
#             'exp_path': '/home/qwang/rob/src/cluster/exps',
#             'exp_name': 'cnn',
#             'rob_name': 'size',
#             
#             'args_json_path': None,
#             'embed_path': '/media/mynewdrive/rob/wordvec/wikipedia-pubmed-and-PMC-w2v.txt',
#             'data_json_path': '/media/mynewdrive/rob/data/rob_tokens.json', #'/home/qwang/rob/amazon_tokens.json',
#             'use_cuda': False,
#             
#             'net_type': 'cnn',
#             'word_hidden_dim': 32,
#             'word_num_layers': 1,
#             
#             'sent_hidden_dim': 32,
#             'sent_num_layers': 1,
#             'max_sent_len': 0,
#             'max_doc_len': 0
#             }
#
#helper = DataIterators(args_dict = args_dict)
## Generate train/valid/test.json
#helper.split_and_save()
#train_data, valid_data, test_data = helper.create_data()   
#train_iterator, valid_iterator, test_iterator = helper.create_iterators(train_data, valid_data, test_data)

#print(helper.LABEL.vocab.stoi)  # {0: 0, 1: 1} ~= {'No': 0, 'Yes': 1}
#helper.TEXT.vocab.itos[:5]  # ['<unk>', '<pad>', 'the', 'i', 'and']
#helper.ID
#
#len(helper.TEXT.vocab)  # 611
#len(helper.LABEL.vocab)  # 2
#
#helper.TEXT.pad_token  # '<pad>'
#helper.TEXT.unk_token  # '<unk>'
#helper.TEXT.vocab.stoi[helper.TEXT.pad_token]  # 1
#helper.TEXT.vocab.stoi[helper.TEXT.unk_token]  # 0
#helper.TEXT.vocab.vectors.shape  # [611, 20]



#class BatchWrapper:
#    def __init__(self, iterator, x_var, y_var):
#        self.iterator = iterator
#        self.x_var = x_var
#        self.y_var = y_var
#    
#    def __iter__(self):
#        for batch in self.iterator:
#            x = getattr(batch, self.x_var)
#            y = getattr(batch, self.y_var)
#            yield x, y
#            
#    
#train_batch = BatchWrapper(train_iterator, "id", "label")
#valid_batch = BatchWrapper(valid_iterator, "id", "label")
#test_batch = BatchWrapper(test_iterator, "id", "text")
#
## Check goldID of first record in first batch in test_iterator
#x_id, y = next(train_batch.__iter__()); print(x_id)
#x_id, y = next(valid_batch.__iter__()); print(x_id)
#x_id, y = next(test_batch.__iter__()); print(x_id); print(x_id[0][0]) # 2314
#
#helper.ID.vocab.itos[x_id[0][0]]  # npqip118
#
## Check goldID of first record in test_list
#data_json_path = args_dict['data_json_path']   
#dat = []
#with open(data_json_path, 'r') as fin:
#    for line in fin:
#        dat.append(json.loads(line))   
#
#random.seed(args_dict['seed'])
#random.shuffle(dat)
#    
#train_size = int(len(dat) * args_dict['train_ratio'])
#val_size = int(len(dat) * args_dict['val_ratio'])
#
#train_list = dat[:train_size]
#val_list = dat[train_size : (train_size + val_size)]
#test_list = dat[(train_size + val_size):]
#test_list[0]['goldID']
#        
#r = 0     
#b = 0
#for g in test_list:
#    r = r + g['RandomizationTreatmentControl']
#    b = b + g['BlindedOutcomeAssessment']
#   
#print(r,b)        
#max_doc_len_list = []
#max_sent_len_list = []
#for i in range(len(train_iterator)): 
#    x_sent, y = next(train_batch.__iter__())
#    x_sent.shape  # [20, 9 ,25] => [batch_size, max_doc_len, max_sent_len]
#    max_doc_len_list.append(x_sent.shape[1])
#    max_sent_len_list.append(x_sent.shape[2])
#    print(i)

  
            
#
#import numpy as np
#print(max(max_doc_len_list), min(max_doc_len_list), np.mean(max_doc_len_list))     # 564, 255, 371.6    
#print(max(max_sent_len_list), min(max_sent_len_list), np.mean(max_sent_len_list))  # 1684, 182, 424.3     
#
#
#d0_tokens = x_sent.permute(1,0)[0]
#s0_tokens = x_sent[0]   # 9 sents in doc 0. Each sent has 25 words
#print(len(x_sent[0][0]))  # sent 0 in doc 0
#
#
##%% NestedField
#import pprint
##from torchtext import data
#pp = pprint.PrettyPrinter(indent=4)
#
#minibatch = [
#     [['he', 'wants', 'a', 'banana'], ['I', 'am', 'sleepy'], ['hello']],
#     [['good'], ['hey', 'how', 'are', 'you']]
#]
## batch_size = 2
##   [doc 1]: doc_len = 3, sent_len = [4,3,1]
##   [doc 2]: doc_len = 2, sent_len = [1,4]
#
#nesting_field = data.Field(pad_token='<pad>', fix_length=None)  # fix num of words in each sent (fix sent_len)
#field = data.NestedField(nesting_field, fix_length=None)  # fix num of sents (fix doc_len)
#padded = field.pad(minibatch) 
#print(len(padded), len(padded[0]), len(padded[0][0])) # batch_size = 2, max_doc_len = 3, max_sent_len = 4
## >> Output
##    [[['he', 'wants', 'a', 'banana'],
##      ['I', 'am', 'sleepy', '<pad>'],
##      ['hello', '<pad>', '<pad>', '<pad>']],
##     [['good', '<pad>', '<pad>', '<pad>'],
##      ['hey', 'how', 'are', 'you'],
##      ['<pad>', '<pad>', '<pad>', '<pad>']]]
#
#
#nesting_field = data.Field(pad_token='<pad>', fix_length=3)  # fix num of words in each sent (fix sent_len)
#field = data.NestedField(nesting_field, fix_length=2)  # fix num of sents (fix doc_len)
#padded = field.pad(minibatch) 
#print(len(padded), len(padded[0]), len(padded[0][0])) # batch_size = 2, max_doc_len = 2, max_sent_len = 3
## >> Output
##    [[['he', 'wants', 'a'], 
##      ['I', 'am', 'sleepy']],
##     [['good', '<pad>', '<pad>'], 
##      ['hey', 'how', 'are']]]


