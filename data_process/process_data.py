#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 17:12:14 2020

@author: qwang
"""

import os
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np

# Change to src dir
src_dir = '/home/qwang/rob/'
os.chdir(src_dir)
from src.data_process.df2json import df2json

# Change to data dir
data_dir = '/media/mynewdrive/rob/'
os.chdir(data_dir)

#%% Read data from different projects
column_list = ['goldID', 'fileLink', 'DocumentLink', 'txtLink',
               'RandomizationTreatmentControl',
               'AllocationConcealment',
               'BlindedOutcomeAssessment',
               'SampleSizeCalculation',
               'AnimalWelfareRegulations',
               'ConflictsOfInterest',
               'AnimalExclusions']

# Stroke
stroke = pd.read_csv("data/stroke/rob_stroke_info.txt", sep='\t', engine="python", encoding="utf-8", index_col = 0)   
df_stroke = stroke[column_list]

# Neuropathic pain
neuro = pd.read_csv("data/np/rob_np_info.txt", sep='\t', engine="python", encoding="utf-8", index_col = 0)   
df_np = neuro[column_list]

# Psychosis
psy = pd.read_csv("data/psycho/rob_psycho_info.txt", sep='\t', engine="python", encoding="utf-8", index_col = 0)   
df_psy = psy[column_list]

# NPQIP
npqip = pd.read_csv("data/npqip/rob_npqip_info.txt", sep='\t', engine="python", encoding="utf-8", index_col = 0)   
npqip['AllocationConcealment'] = float('nan')
npqip['AnimalWelfareRegulations'] = float('nan')
npqip['ConflictsOfInterest'] = float('nan')
npqip['AnimalExclusions'] = float('nan')
df_npqip = npqip[column_list]

# IICARus
iicarus = pd.read_csv("data/iicarus/rob_iicarus_info.txt", sep='\t', engine="python", encoding="utf-8", index_col = 0)   
iicarus['AllocationConcealment'] = float('nan')
iicarus['AnimalWelfareRegulations'] = float('nan')
iicarus['ConflictsOfInterest'] = float('nan')
iicarus['AnimalExclusions'] = float('nan')
df_iicarus = iicarus[column_list]

# Concatenate dataframe
frames = [df_stroke, df_np, df_psy, df_npqip, df_iicarus]
df = pd.concat(frames)

#%% Tokenization
df2json(df_info = df, json_path = 'data/rob_word_sent_tokens.json')


#%% Check number of words and sents
#def read_json(json_path):
#    df = []
#    with open(json_path, 'r') as fin:
#        for line in fin:
#            df.append(json.loads(line))
#    return df
#
#df = read_json(json_path='data/rob_fulltokens.json')
#
#goldID_del = []
#num_w = []
#num_s = []
#
#for g in df:
#    if len(g['wordTokens']) < 1000 or len(g['wordTokens']) > 10000:
#        goldID_del.append(g['goldID'])
#    else:
#        num_w.append(len(g['wordTokens']))
#
#for g in df:
#    if len(g['sentTokens']) < 20:
#        goldID_del.append(g['goldID'])
#    else:
#        num_s.append(len(g['sentTokens']))
#        
#        
##gold_final = [g for g in df if g['goldID'] not in goldID_del]  # 7877
#
#print(max(num_w), min(num_w), np.mean(num_w))  # 18734, 1008, 5508
#print(max(num_s), min(num_s), np.mean(num_s))  # 2680, 20, 219
#
#
## Histogram for tokens
#plt.hist(num_w, bins=40, edgecolor='black', alpha=0.8)
#plt.xlabel("Number of word tokens")
#plt.ylabel("Frequency")
#plt.show()
#
#plt.hist(num_s, bins=40, edgecolor='black', alpha=0.8)
#plt.xlabel("Number of sent tokens")
#plt.ylabel("Frequency")
#plt.show()

#%%
