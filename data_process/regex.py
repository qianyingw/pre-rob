#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 16:28:49 2019

@author: qwang
"""

import re
import pickle
import pandas as pd
import os
from tqdm import tqdm
data_dir = '/media/mynewdrive/rob/'
os.chdir(data_dir)

#%%
# Read regex string
def read_regex(regex_path):
    with open(regex_path, "r", encoding='utf-8') as fin:
        regex = fin.read()
    return regex

regex_random = read_regex('regex/RandomizationTreatmentControl.txt')
regex_blind = read_regex('regex/BlindedOutcomeAssessment.txt')
regex_sample = read_regex('regex/SampleSizeCalculation.txt')
regex_conflict = read_regex('regex/ConflictsOfInterest.txt')
regex_welfare = read_regex('regex/AnimalWelfareRegulations.txt')


# Document annotation
def doc_annotate(regex, doc):
    match = re.search(regex, doc)
    if match:
        doc_label = 1
    else:
        doc_label = 0
    return doc_label



#%%
# Sentence tokenisation and annotation
#def sent_annotate(regex, doc):
#    sent_label = []
#    sent_list = nltk.sent_tokenize(doc)
#    for i, sent in enumerate(sent_list):
#        match = re.search(regex, sent)
#        if match:
#            sent_label.append([sent, 1])
#        else:
#            sent_label.append([sent, 0])
#    return sent_label

##%% test
#test_str = "When calculating the average latency, the cut-off time was assigned to the normal responses. The average latency was taken as ameasure for the severity of cold allodynia; shorter tail withdrawal latency was interpreted as more severe allodynia. All behavioral tests were performed in blinded fashion."
#sent_labeled = sent_annotate(regex_blinding, test_str)
#sent_labeled[0]    # [sentence, label]
#sent_labeled[0][0] # sentence
#sent_labeled[0][1] # label

#%% Error checking for gold data
df = pd.read_pickle('data/rob_info_a.pkl')  # for random/blind/sample
 
# Get regex label
for i, row in tqdm(df.iterrows()): 
    # Read string list from pkl file
    pkl_path = os.path.join('data/rob_str', df.loc[i,'goldID']+'.pkl') 
    with open(pkl_path, 'rb') as fin:
        sent_ls = pickle.load(fin)
        
    # Extract text    
    text = ''
    for l in sent_ls:
        t = ' '.join(l)
        text = text + t       
    df.loc[i,'rgx_random'] = doc_annotate(regex_random, text)
    df.loc[i,'rgx_blind'] = doc_annotate(regex_blind, text)
    df.loc[i,'rgx_sample'] = doc_annotate(regex_sample, text)
    df.loc[i,'rgx_conflict'] = doc_annotate(regex_conflict, text)
    df.loc[i,'rgx_welfare'] = doc_annotate(regex_welfare, text)

list(df.columns)

# Mismatch
for i in range(len(df)):
    df.loc[i, 'project'] = re.sub(r'\d+', "", df.loc[i, 'goldID'])

sub = df
sub = df[df.project == 'iicarus']
df_mis = sub[sub.rgx_random != sub.RandomizationTreatmentControl]; print(len(df_mis)) 
df_mis = sub[sub.rgx_blind != sub.BlindedOutcomeAssessment]; print(len(df_mis)) 
df_mis = sub[sub.rgx_sample != sub.SampleSizeCalculation]; print(len(df_mis)) 
df_mis = sub[(sub['ConflictsOfInterest'].isnull() == False) & (sub.rgx_conflict != sub.ConflictsOfInterest)]; print(len(df_mis)) 
df_mis = sub[(sub['AnimalWelfareRegulations'].isnull() == False) & (sub.rgx_welfare != sub.AnimalWelfareRegulations)]; print(len(df_mis)) 

df_mis = sub[(sub.rgx_random != sub.RandomizationTreatmentControl) | 
             (sub.rgx_blind != sub.BlindedOutcomeAssessment) |
             (sub.rgx_sample != sub.SampleSizeCalculation) |
             ((sub['ConflictsOfInterest'].isnull() == False) & (sub.rgx_conflict != sub.ConflictsOfInterest)) |
             ((sub['AnimalWelfareRegulations'].isnull() == False) & (sub.rgx_welfare != sub.AnimalWelfareRegulations))]
