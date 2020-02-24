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
#iicarus['AllocationConcealment'] = float('nan')
#iicarus['AnimalWelfareRegulations'] = float('nan')
#iicarus['ConflictsOfInterest'] = float('nan')
#iicarus['AnimalExclusions'] = float('nan')
df_iicarus = iicarus[column_list]

# Concatenate dataframe
frames = [df_stroke, df_np, df_psy, df_npqip, df_iicarus]
df = pd.concat(frames)

#%% Tokenization
df2json(df_info = df[:1500], json_path = 'data/rob1.json')
df2json(df_info = df[1500:3000], json_path = 'data/rob2.json')
df2json(df_info = df[3000:4500], json_path = 'data/rob3.json')
df2json(df_info = df[4500:6000], json_path = 'data/rob4.json')
df2json(df_info = df[6000:7000], json_path = 'data/rob5.json')
df2json(df_info = df[7000:], json_path = 'data/rob6.json')


def read_json(json_path):
    df = []
    with open(json_path, 'r') as fin:
        for line in fin:
            df.append(json.loads(line))
    return df

rob1 = read_json(json_path='data/rob1.json')
rob2 = read_json(json_path='data/rob2.json')
rob3 = read_json(json_path='data/rob3.json')
rob4 = read_json(json_path='data/rob4.json')
rob5 = read_json(json_path='data/rob5.json')
rob6 = read_json(json_path='data/rob6.json')

gold = rob1 + rob2 + rob3 + rob4 + rob5 + rob6

# Merge json files
#gold = read_json(json_path='data/rob1.json')
#for i in range(9):
#    rob = read_json(json_path='data/rob'+str(i+2)+'.json')
#    gold = gold + rob
 
# Or tokenize together...
# df2json(df_info = df, json_path = 'data/rob_word_sent_tokens.json')
# df = read_json(json_path='data/rob_word_sent_tokens.json')

#%% Output
# Check number of words and sents
goldID_del = []  
num_w, num_s = [], []      
for g in gold:
    if len(g['wordTokens']) < 1000 or len(g['sentTokens']) < 50:
        goldID_del.append(g['goldID'])
    else:
        num_w.append(len(g['wordTokens']))
        num_s.append(len(g['sentTokens']))

print(max(num_w), min(num_w), np.mean(num_w))  # 15889, 1125, 5000
print(max(num_s), min(num_s), np.mean(num_s))  # 686, 50, 181        
        
gold_final = [g for g in gold if g['goldID'] not in goldID_del]  # 7840
with open('data/rob_tokens.json', 'w') as fout:
    for g in gold_final:     
        fout.write(json.dumps(g) + '\n')

# Output info file
gold_info = df[-df["goldID"].isin(goldID_del)]
gold_info.to_csv('data/rob_info.txt', sep='\t', encoding='utf-8')
list(gold_info.columns)


#%% Insert annotations for iicarus/npqip
gold_final = read_json(json_path='data/rob_tokens.json')

for idx, row in df_iicarus.iterrows():
    try:
        dct = next(item for item in gold_final if item["goldID"] == row['goldID'])
        dct['AllocationConcealment'] = row['AllocationConcealment']
        dct['AnimalWelfareRegulations'] = row['AnimalWelfareRegulations']
        dct['ConflictsOfInterest'] = row['ConflictsOfInterest']
        dct['AnimalExclusions'] = row['AnimalExclusions']
    except:
        print('Not in final gold data: {}'.format(row['goldID']))

# Check if replacementis done
for g in gold_final:
    if g['goldID'][:7] == 'iicarus':
        print(g['AllocationConcealment'], g['AnimalWelfareRegulations'], g['ConflictsOfInterest'], g['AnimalExclusions'])
        
# Output rob_tokens.json and rob_tokens_sub.json (no npqip)
with open('data/rob_tokens.json', 'w') as fout:
    for g in gold_final:     
        fout.write(json.dumps(g) + '\n')  # 7840

with open('data/rob_tokens_sub.json', 'w') as fout:
    for g in gold_final:   
        if g['goldID'][:5] != 'npqip':
            fout.write(json.dumps(g) + '\n')  # 7089

#%% Histogram for tokens
plt.hist(num_w, bins=40, edgecolor='black', alpha=0.8)
plt.xlabel("Number of word tokens")
plt.ylabel("Frequency")
plt.show()

plt.hist(num_s, bins=40, edgecolor='black', alpha=0.8)
plt.xlabel("Number of sent tokens")
plt.ylabel("Frequency")
plt.show()


