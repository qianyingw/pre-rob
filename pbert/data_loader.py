#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 17:23:24 2020

@author: qwang
"""


import os
import pickle
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import sklearn

import transformers
   
#%%
class RobDataset(Dataset):
    """
    info_file: 
        'rob_info_a.pkl' (for RandomizationTreatmentControl | BlindedOutcomeAssessment | SampleSizeCalculation | AnimalExclusions)     
        'rob_info_b.pkl' (for AllocationConcealment | AnimalWelfareRegulations | ConflictsOfInterest)   
    group: 'train', 'valid', 'test'  
    
    Returns:
        doc: [num_chunks, 3, max_chunk_len]. '3' refers to tokens_ids, attn_masks, token_type_ids
        label
        
    """
    def __init__(self, info_dir, pkl_dir, rob_item, max_chunk_len, max_n_chunk, group, tokenizer):  
        
       
        
        if rob_item in ['RandomizationTreatmentControl','BlindedOutcomeAssessment','SampleSizeCalculation','AnimalExclusions']:
            info_file = os.path.join(info_dir, 'rob_info_a.pkl')
        else:  # ['AllocationConcealment','AnimalWelfareRegulations','ConflictsOfInterest'] 
            info_file = os.path.join(info_dir, 'rob_info_b.pkl')
            
        info_df = pd.read_pickle(info_file)
        
        # ########## test only ########## 
        # info_df = sklearn.utils.shuffle(info_df, random_state=1234)
        # info_df = info_df[:20]
        # ###############################    
        if group == 'train':
            print('Overall data size: {}'.format(len(info_df)))
               
        if group:
            info_df = info_df[info_df['partition']==group]
        self.info_df = info_df.reset_index(drop=True)
        
        self.pkl_dir = pkl_dir
        self.rob_item = rob_item   
        self.max_chunk_len = max_chunk_len
        self.max_n_chunk = max_n_chunk
        self.tokenizer = tokenizer
        
        
    def __len__(self):
        return len(self.info_df)
    
    def __getitem__(self, idx):
        
        pkl_path = os.path.join(self.pkl_dir, self.info_df.loc[idx, 'goldID']+'.pkl') 
        with open(pkl_path, 'rb') as fin:
            sent_ls = pickle.load(fin)
        
        ###### BERT input tokenization and formatting ######
        if type(self.tokenizer) == transformers.tokenization_bert.BertTokenizer:     
            # Convert sent list back to plain text
            sent_ls = [s[0] for s in sent_ls]
            text = ". ".join(sent_ls)  
            # Convert text to tokens by WordPiece
            tokens = self.tokenizer.tokenize(text)
            
            # Split tokens into chunks
            n_chunks = len(tokens) // (self.max_chunk_len - 2)
            if len(tokens) % (self.max_chunk_len - 2) != 0:
                n_chunks += 1           
            
            # Limit number of chunks
            if n_chunks > self.max_n_chunk:
                tokens = tokens[:self.max_chunk_len*self.max_n_chunk]
                n_chunks = self.max_n_chunk  
            assert n_chunks <= self.max_n_chunk, "The document is too large. Try to increase cut_head/tail_ratio."
            
            # Document tensor
            doc = torch.zeros((n_chunks, 3, self.max_chunk_len), dtype=torch.long)
            for i in range(n_chunks):
                chunk_tokens = tokens[(self.max_chunk_len-2) * i : (self.max_chunk_len-2) * (i+1)]
                chunk_tokens.insert(0, "[CLS]")
                chunk_tokens.append("[SEP]")              
                chunk_tokens_ids = self.tokenizer.convert_tokens_to_ids(chunk_tokens)
                      
                attn_masks = [1] * len(chunk_tokens_ids)         
                # Pad the last chunk
                while len(chunk_tokens_ids) < self.max_chunk_len:
                    chunk_tokens_ids.append(0)
                    attn_masks.append(0)
                    
                token_type_ids = [0] * self.max_chunk_len       
                assert len(chunk_tokens_ids) == self.max_chunk_len and len(attn_masks) == self.max_chunk_len
                             
                doc[i] = torch.cat((torch.LongTensor(chunk_tokens_ids).unsqueeze(0),
                                    torch.LongTensor(attn_masks).unsqueeze(0),
                                    torch.LongTensor(token_type_ids).unsqueeze(0)), dim=0)
        
            
        # Label tensor
        label = self.info_df.loc[idx, self.rob_item]
        label = torch.LongTensor([label])  
                
        return doc, label
    
    def cls_weight(self):
        df = self.info_df   
        n_pos = len(df[df[self.rob_item]==1])
        n_neg = len(df[df[self.rob_item]==0])   
        return [1/n_neg, 1/n_pos]  # [1/n_pos, 1/n_neg]
        

class PadDoc:
    def __call__(self, batch):
        # Element in batch: (doc, label)
        # Sort batch by num_chunks in descending order. x[0] => doc, x[1] -> label
        # x[0].shape
        #       BERT: [num_chunks, 3, max_chunk_len]
        #       XLNet: [n_tokens, 2]
        #       Longformer: [n_tokens, 3]
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)  # 
        
        # Pad doc within batch       		
        docs = [x[0] for x in sorted_batch]
        docs_padded = nn.utils.rnn.pad_sequence(docs, batch_first=True)
        
		# Obtain labels of sorted batch
        labels = torch.LongTensor(list(map(lambda x: x[1], sorted_batch)))

        return docs_padded, labels


      
#%% Instance
# data_set = RobDataset(info_dir = '/media/mynewdrive/rob/data',
#                       pkl_dir = '/media/mynewdrive/rob/data/rob_str',
#                       rob_item = 'RandomizationTreatmentControl',
#                       max_chunk_len = 512, 
#                       max_n_chunk = 10000,
#                       group = 'train',
#                       tokenizer=tokenizer)  # 'valid', 'test'        


# len(data_set)  # 6272
# data_set.cls_weight()
# idx = 0
# doc = data_set[idx][0]
# doc[0]   # 1st chunk
# doc[-1]  # last chunk


# n_chk = []
# for i in range(len(data_set)):
#     output = data_set[i][0]
#     n_chk.append(output.size()[0])
#     if i % 1000 == 0:
#         print(output.size())

# max num_chunk in 23 (train), 21 (valid), 25 (test)
# max(n_chk)  


# # DataLoader
# from torch.utils.data import DataLoader
# data_loader = DataLoader(data_set, batch_size=4, shuffle=True, num_workers=4, collate_fn=PadDoc())

# batch = next(iter(data_loader))
# doc_batch = batch[0] 
# label_batch = batch[1]   

# doc_batch.size()  # [batch_size, num_chunks, 3, max_chunk_len]
# label_batch.size()  # [batch_size]

# for i, batch in enumerate(data_loader):
#     if i % 50 == 0:
#         print("[batch {}] Doc: {}, Label: {}".format(i, batch[0].size(), batch[1].size()))