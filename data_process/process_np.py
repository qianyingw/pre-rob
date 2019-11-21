#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 12:14:08 2019

@author: qwang
"""

import os

import pandas as pd
import numpy as np

import shutil

# Change to src dir
src_dir = '/home/qwang/rob/'
os.chdir(src_dir)
from src.data_process.df2json import df2json

# Change to data dir
data_dir = '/media/mynewdrive/rob/'
os.chdir(data_dir)

#%% Read and format data
neuro = pd.read_csv("data/np/NP_RawFinal.csv", sep=',', engine="python")   
list(neuro.columns)
# Change column name
neuro = neuro.rename(columns={'Randomisation': 'RandomizationTreatmentControl'})
neuro['ID'] = np.arange(1, len(neuro)+1)

    
    
#%% Copy related PDFs to a separate folder
#root_path = 'U:/Datastore/CMVM/scs/groups/DCN/TRIALDEV/CAMARADES'
# Remove records without DocumentLink or fileLink
neuro = neuro.dropna(subset=['DocumentLink', 'fileLink'], how='all')  # len(neuro) = 1794
# Remove record with invalid link
neuro = neuro[-neuro["PubID"].isin([29781])]  # len(neuro) = 1793

# Manually correct document links
neuro.loc[neuro.PubID==20794, 'DocumentLink'] = 'Qianying/rob/data/np/PDFs_manual/4368_Palazzo.pdf'
neuro.loc[neuro.PubID==20860, 'DocumentLink'] = 'Qianying/rob/data/np/PDFs_manual/957_Crown_2012.pdf'
neuro.loc[neuro.PubID==24208, 'DocumentLink'] = 'Qianying/rob/data/np/PDFs_manual/Tan.pdf'
neuro.loc[neuro.PubID==36558, 'DocumentLink'] = 'Qianying/rob/data/np/PDFs_manual/23991_Miller Hoschouer_2008.pdf'
neuro.loc[neuro.PubID==400226, 'DocumentLink'] = 'Qianying/rob/data/np/PDFs_manual/400226_Ahn_2014.pdf'

neuro.loc[neuro.PubID==402532, 'DocumentLink'] = 'Qianying/rob/data/np/PDFs_manual/402532_Chen_2014.pdf'
neuro.loc[neuro.PubID==403840, 'DocumentLink'] = 'Qianying/rob/data/np/PDFs_manual/Di Cesare Manne-2015-Widespread pain reliever.pdf'
neuro.loc[neuro.PubID==404641, 'DocumentLink'] = 'Qianying/rob/data/np/PDFs_manual/Fariello-2014-Broad spectrum and prolonged eff.pdf'
neuro.loc[neuro.PubID==408349, 'DocumentLink'] = 'Qianying/rob/data/np/PDFs_manual/408349_Kato_2014.pdf'
neuro.loc[neuro.PubID==409788, 'DocumentLink'] = 'Qianying/rob/data/np/PDFs_manual/Lee-2013-Analgesic effect of acupuncture is me.pdf'

neuro.loc[neuro.PubID==413288, 'DocumentLink'] = 'Qianying/rob/data/np/PDFs_manual/Novrup-2014-Central but not systemic administr.pdf'
neuro.loc[neuro.PubID==417813, 'DocumentLink'] = 'Qianying/rob/data/np/PDFs_manual/tnsci-2015-0010.pdf'
neuro.loc[neuro.PubID==420986, 'DocumentLink'] = 'Qianying/rob/data/np/PDFs_manual/420986_Zhao_2014.pdf'


# Copy files
neuro['DocumentLink_old'] = neuro['DocumentLink'].str.replace('\\', "/")
for i, row in neuro.iterrows():     
#    doclink = neuro.loc[i,'DocumentLink'].split(";")[0]
#    if len(os.path.basename(doclink).split('.pdf')) == 1:
#        doclink = doclink + '.pdf'

#    old_path = os.path.join(root_path, doclink)  
    new_path = os.path.join('data/np/PDFs', 'np_'+str(i+1)+'.pdf')
    neuro.loc[i,'DocumentLink'] = new_path    
#    if os.path.exists(old_path) == False:
#        print('PDF does not exist: {}'.format(old_path))
#    else:
#        shutil.copy2(old_path, new_path)


# Modify txt paths
neuro['txtLink'] = neuro['DocumentLink'].str.replace('.pdf', ".txt") 
neuro['txtLink'] = neuro['txtLink'].str.replace('PDFs', "TXTs")


#%% Convert pdf to txt by Xpdf
# See pdf2txt_np.sh

#%% Check existence of txt files
ID_del1 = []
link_del1 = []
for i, row in neuro.iterrows():      
    txt_path = neuro.loc[i,'txtLink']  
    if os.path.exists(txt_path):
        with open(txt_path, 'r', encoding='utf-8') as fp:
            text = fp.read()
        neuro.loc[i,'textLen'] = len(text)
    else:
        neuro.loc[i,'textLen'] = 0
        
    if neuro.loc[i,'textLen'] < 1000:
        ID_del1.append(neuro.loc[i, 'ID'])
        link_del1.append(neuro.loc[i, 'DocumentLink'])

      
# Remove records which do not exist or too short
neuro = neuro[-neuro["ID"].isin(ID_del1)]  # len(neuro) = 1694

#%% Check long and short texts (not from csv file)
print(max(neuro['textLen']))  # 1931095
print(min(neuro['textLen']))  # 2130

ID_del2 = []
for _, row in neuro.iterrows():   
    if (row['textLen'] > 100000) or (row['textLen'] < 9000):
        ID_del2.append(row['ID'])
print('IDs of txts with too long or short texts:\n')
print(ID_del2)
# [102, 131, 219, 295, 682, 732, 771, 858, 860, 866, 904, 948, 951, 964, 966, 967, 981, 1049, 1085, 1095, 1549]

temp = neuro[neuro["ID"].isin(ID_del2)]
temp = temp[['ID', 'textLen', 'DocumentLink']]


# Records with too short/long texts need to be removed (manually checked; two records were removed)
neuro = neuro[-neuro["ID"].isin([948,904,951,964,682,
                                 732,981,1095,295,1085,
                                 967,966,1549,1049,866,771])]  # len(neuro) = 1678

# Re-index
neuro.set_index(pd.Series(range(0, len(neuro))), inplace=True)

#%% Remove records with unique length
np_dup = neuro[neuro.duplicated(subset=['textLen'], keep=False)]
np_dup.loc[:,'textSame'] = ''
# Check whether records with same text length are duplicate
dup_grouped = np_dup.groupby(['textLen'])
dup_grouped = list(dup_grouped)
len(dup_grouped)  # 71

duplen = []  
for i in range(len(dup_grouped)):
    duplen.append(len(dup_grouped[i][1]))
set(duplen)  # {2, 3}

for i, tup in enumerate(dup_grouped):
    if len(tup[1]) == 2:
        df = tup[1]
        df.set_index(pd.Series(range(0,len(df))), inplace=True)
        path0 = df['txtLink'][0]
        path1 = df['txtLink'][1]
        with open(path0, 'r', encoding='utf-8') as fp:
            text0 = fp.read()
        with open(path1, 'r', encoding='utf-8') as fp:
            text1 = fp.read()      
        if text0 == text1:
            df['textSame'][0] = 'Yes'

# Convert list to dataframe
frames = [dg[1] for dg in dup_grouped]
dup_df = pd.concat(frames)

dup_df.to_csv('data/np/np_duplicates.csv', sep=',', encoding='utf-8')


# Remove duplicate records
ID_del3 = [1308,678,1571,459,699,
            1222,659,539,1354,676,
            1185,453,1197,1232,268,
            1313,417,1574,163,333,
            33,180,91,1196,316,
            185,391,1246,1568,1141,
            1137,824]
neuro = neuro[-neuro["ID"].isin(ID_del3)]  # len(neuro) = 1646   


# Manually correct labels by combine labels from single-extracted and dual-extracted
# (only for randomisation/blinded/ssz)
# See 'np_duplicates.xlsx' for final IDs needs to be removed/corrected
# Can't have any decision so just delete all 20 duplicate records with different labels
ID_del4 = [380,570,303,630,816,
            22,154,1555,930,14,
            273,321,1569,29,297,
            476,93,1123,929,1572]

neuro = neuro[-neuro["ID"].isin(ID_del4)]  # len(neuro) = 1626


# Re-index
neuro.set_index(pd.Series(range(0, len(neuro))), inplace=True)


#%% Output data
# Add columns
neuro['goldID'] = 'np' + neuro['ID'].astype(str)  # ID for all the gold data
neuro = neuro.dropna(subset=['RandomizationTreatmentControl', 'BlindedOutcomeAssessment', 'SampleSizeCalculation'], how='all')
neuro = neuro.dropna(subset=['RandomizationTreatmentControl', 'BlindedOutcomeAssessment', 'SampleSizeCalculation'], how='any')

# Replace TRUE/FALSE by 1/0
neuro.RandomizationTreatmentControl = neuro.RandomizationTreatmentControl.astype(int)
neuro.AllocationConcealment = neuro.AllocationConcealment.astype(int)
neuro.BlindedOutcomeAssessment = neuro.BlindedOutcomeAssessment.astype(int)
neuro.SampleSizeCalculation = neuro.SampleSizeCalculation.astype(int)
neuro.AnimalWelfareRegulations = neuro.AnimalWelfareRegulations.astype(int)
neuro.ConflictsOfInterest = neuro.ConflictsOfInterest.astype(int)
neuro.AnimalExclusions = neuro.AnimalExclusions.astype(int)

neuro.to_csv('data/np/rob_np_info.txt', sep='\t', encoding='utf-8')
list(neuro.columns)

#['PubID',
# 'Surnames',
# 'Years',
# 'PublicationIDs',
# 'RandomizationTreatmentControl',
# 'AllocationConcealment',
# 'BlindedOutcomeAssessment',
# 'SampleSizeCalculation',
# 'AnimalWelfareRegulations',
# 'ConflictsOfInterest',
# 'AnimalExclusions',
# 'DocumentLink',
# 'fileLink',
# 'source',
# 'ID',
# 'DocumentLink_old',
# 'txtLink',
# 'textLen',
# 'goldID']



#%% Tokenization to json file
neuro = pd.read_csv("data/np/rob_np_info.txt", sep='\t', engine="python", encoding="utf-8", index_col = 0)   

df = neuro[['goldID',
              'fileLink',
              'DocumentLink',
              'txtLink',
              'RandomizationTreatmentControl',
              'AllocationConcealment',
              'BlindedOutcomeAssessment',
              'SampleSizeCalculation',
              'AnimalWelfareRegulations',
              'ConflictsOfInterest',
              'AnimalExclusions']]

df2json(df_info = df, json_path = 'data/np/rob_np_fulltokens.json')
