#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 14:49:20 2020

@author: qwang
"""

import os
import pickle
import pandas as pd

import torch
from torch.utils.data import Dataset

# import transformers
from transformers import BertTokenizer
from sentence_transformers import SentenceTransformer, util
sbert_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#%%
class RobDataset(Dataset):
    """
    info_file: 
        'rob_info_a.pkl' (for RandomizationTreatmentControl | BlindedOutcomeAssessment | SampleSizeCalculation | AnimalExclusions)     
        'rob_info_b.pkl' (for AllocationConcealment | AnimalWelfareRegulations | ConflictsOfInterest)   
    group: 'train', 'valid', 'test'  
    
    Returns:
        doc: [3, max_chunk_len]. '3' refers to tokens_ids, attn_masks, token_type_ids
        label
    """
    
    def __init__(self, info_dir, pkl_dir, rob_item, rob_sent, max_n_sent, group):
        
        
        if rob_item in ['RandomizationTreatmentControl','BlindedOutcomeAssessment','SampleSizeCalculation','AnimalExclusions']:
            info_file = os.path.join(info_dir, 'rob_info_a.pkl')
        else:  # ['AllocationConcealment','AnimalWelfareRegulations','ConflictsOfInterest'] 
            info_file = os.path.join(info_dir, 'rob_info_b.pkl')
        
        if rob_sent is None:
            if rob_item == 'RandomizationTreatmentControl':
                self.rob_sent = 'Animals are randomly allocated to treatment or control groups at the start of the experimental treatment'
            if rob_item == 'BlindedOutcomeAssessment':
                self.rob_sent = 'Assessment of an outcome in a blinded fashion. Investigators measuring the outcome do not know which treatment group the animals belongs to and what treatment they had received' 
            if rob_item == 'SampleSizeCalculation':
                self.rob_sent = 'The manuscript reports the performance of a sample size calculation and describes how this number was derived statistically' 
            if rob_item == 'AnimalExclusions':
                self.rob_sent = 'All animals, all data and all outcomes measured are accounted for and presented in the final analysis. Reasons are given for animal exclusions' 
            if rob_item == 'AllocationConcealment':
                self.rob_sent = 'Investigators performing the experiment do not know which treatment an animal is being given' 
            if rob_item == 'AnimalWelfareRegulations':
                self.rob_sent = 'Research investigators complied with animal welfare regulations' 
            if rob_item == 'ConflictsOfInterest':
                self.rob_sent = 'Potential conflict of interest, like funding or affiliation to a pharmaceutical company' 
                               
        info_df = pd.read_pickle(info_file)
        
        if group:
            info_df = info_df[info_df['partition']==group]
        self.info_df = info_df.reset_index(drop=True)
        
        self.info_dir = info_dir  
        self.pkl_dir = pkl_dir
        self.rob_item = rob_item
        self.max_n_sent = max_n_sent

    def __len__(self):
        return len(self.info_df)
    
    def __getitem__(self, idx):
        
        pkl_path = os.path.join(self.pkl_dir, self.info_df.loc[idx, 'goldID']+'.pkl') 
        with open(pkl_path, 'rb') as fin:
            sents = pickle.load(fin)
        sents = [s[0] for s in sents] 
            
        sent_embeds = sbert_model.encode(sents, convert_to_tensor=True)
        rob_embed = sbert_model.encode([self.rob_sent], convert_to_tensor=True)
        
        # Compute cosine-similarities for rob sentence with each sentence in fulltext
        cos = util.pytorch_cos_sim(sent_embeds, rob_embed)
        
        # Find the pairs with the highest cosine similarity scores
        pairs = []
        for i in range(cos.shape[0]):
            pairs.append({'index': i, 'score': cos[i][0]})   
        # Sort scores in decreasing order
        pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)
                
        sim_sents = []
        for pair in pairs[:self.max_n_sent]:
            sim_sents.append(sents[pair['index']])
        
        sim_text = ". ".join(sim_sents)    
            
        label = self.info_df.loc[idx, self.rob_item]
        # label = torch.LongTensor([label])  
               
        return sim_text, label

#%%
class BatchTokenizer:
    def __call__(self, batch):
        # Each item in a batch: (text, label)
        texts, labels = [], []
        for item in batch:
            texts.append(item[0])
            labels.append(item[1])
        
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")    
        # input_ids = inputs['input_ids']
        # token_type_ids = inputs['token_type_ids']
        # attention_mask = inputs['attention_mask']
        
        labels = torch.LongTensor(labels)

        return inputs, labels
    
#%% Instance
train_set = RobDataset(info_dir = '/media/mynewdrive/rob/data', pkl_dir = '/media/mynewdrive/rob/data/rob_str', 
                       rob_item = 'RandomizationTreatmentControl', rob_sent = None, 
                       max_n_sent = 20,
                       group='train')

# DataLoader
from torch.utils.data import DataLoader
data_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0, collate_fn=BatchTokenizer())

batch = next(iter(data_loader))
input_batch = batch[0]; print(input_batch.size())   
label_batch = batch[1]; print(label_batch.size())    
len_batch = batch[2]; print(len_batch)  

doc_batch.size()  # [batch_size, num_chunks, 3, max_chunk_len]
label_batch.size()  # [batch_size]
len_batch.size()  # [batch_size]

for i, batch in enumerate(data_loader):
    if i % 50 == 0:
        print("[batch {}] Doc: {}, Label: {}".format(i, batch[0].size(), batch[1].size()))
temp = train_set[0][0]



