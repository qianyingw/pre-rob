# -*- coding: utf-8 -*-
"""
Clean stroke data on local PC


Created on Tue Nov 19 17:24:10 2019
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
#import csv
#csv.field_size_limit(100000000)
#stroke = pd.read_csv("data/stroke/dataWithFullText_utf8.csv", sep=',', engine="python", encoding="utf-8")   
#list(stroke.columns)
#
## Remove column 'CleanFullText'
#stroke = stroke[['ID',
#                'RandomizationTreatmentControl',
#                'AllocationConcealment',
#                'BlindedOutcomeAssessment',
#                'SampleSizeCalculation',
#                'AnimalExclusions',
#                'Comorbidity',
#                'AnimalWelfareRegulations',
#                'ConflictsOfInterest',
#                'DocumentLink',
#                'fileLink']]
#
#stroke.to_csv('data/stroke/Stroke_RawFinal.csv', sep='\t', encoding='utf-8')


stroke = pd.read_csv("data/stroke/Stroke_RawFinal.csv", sep='\t', engine="python", encoding="utf-8", index_col = 0)   
list(stroke.columns)
# Chancge column names
stroke.columns = ['RecordID',
                  'RandomizationTreatmentControl',
                  'AllocationConcealment',
                  'BlindedOutcomeAssessment',
                  'SampleSizeCalculation',
                  'AnimalExclusions',
                  'Comorbidity',
                  'AnimalWelfareRegulations',
                  'ConflictsOfInterest',
                  'DocumentLink',
                  'fileLink'] 

stroke['ID'] = np.arange(1, len(stroke)+1)

#%% Copy related PDFs to a separate folder
stroke['DocumentLink_old'] = stroke['DocumentLink'].str.replace('\\', "/")
for i, row in stroke.iterrows():  
    old_path = os.path.join('data/stroke/stkPublications', stroke.loc[i,'DocumentLink'])             
    new_path = os.path.join('data/stroke/PDFs', 'stroke_'+str(i+1)+'.pdf')
    stroke.loc[i,'DocumentLink'] = new_path    
#    if os.path.exists(old_path) == False:
#        print('PDF does not exist: {}'.format(old_path))
#    else:
#        shutil.copy2(old_path, new_path)

# No missing PDFs!

# Modify txt paths
stroke['txtLink'] = stroke['DocumentLink'].str.replace('.pdf', ".txt") 
stroke['txtLink'] = stroke['txtLink'].str.replace('PDFs', "TXTs")

#%% Convert pdf to txt by Xpdf
# See pdf2txt_stroke.sh

#%% Check existence of txt files
ID_del1 = []
link_del1 = []
for i, row in stroke.iterrows():      
    txt_path = stroke.loc[i,'txtLink']  
    if os.path.exists(txt_path):
        with open(txt_path, 'r', encoding='utf-8') as fp:
            text = fp.read()
        stroke.loc[i,'textLen'] = len(text)
    else:
        stroke.loc[i,'textLen'] = 0
        
    if stroke.loc[i,'textLen'] < 1000:
        ID_del1.append(stroke.loc[i, 'ID'])
        link_del1.append(stroke.loc[i, 'DocumentLink'])

      
# Remove records which do not exist or too short
stroke = stroke[-stroke["ID"].isin(ID_del1)]  # len(stroke) = 2502
         
#%% Check long and short texts (not from csv file)
print(max(stroke['textLen']))  # 4479878
print(min(stroke['textLen']))  # 10875

ID_del2 = []
for _, row in stroke.iterrows():   
    if (row['textLen'] > 100000) or (row['textLen'] < 9000):
        ID_del2.append(row['ID'])
print('IDs of txts with too long or short texts:\n')
print(ID_del2)
# [75, 620, 726, 1431, 1518, 2009, 2062]

temp = stroke[stroke["ID"].isin(ID_del2)]
temp = temp[['ID', 'textLen', 'DocumentLink']]


# Records with too short/long texts need to be removed (manually checked; two records were removed)
stroke = stroke[-stroke["ID"].isin([75, 620])]  # len(stroke) = 2500

# Re-index
stroke.set_index(pd.Series(range(0, len(stroke))), inplace=True)


#%% Remove records with unique length
stroke_dup = stroke[stroke.duplicated(subset=['textLen'], keep=False)]
stroke_dup.loc[:,'textSame'] = ''
# Check whether records with same text length are duplicate
dup_grouped = stroke_dup.groupby(['textLen'])
dup_grouped = list(dup_grouped)
len(dup_grouped)  # 105

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


dup_df.to_csv('/home/qwang/rob/stroke_duplicates.csv', sep=',', encoding='utf-8')
dup_df.to_csv('data/stroke/stroke_duplicates.csv', sep=',', encoding='utf-8')

# Remove one duplicate records
ID_del3 = [2447,1118,1100,1105,732,
           593,708,710,238,2440,
           1880,2235,97,1090,2445,
           1611,635,2227,2286,209,
           636,2494,58,2283,779,
           1992,760,778,1618]
stroke = stroke[-stroke["ID"].isin(ID_del3)]  # len(stroke) = 2471    


# Manually correct labels based on 'stroke_PublicationData error correction project_fullAnnotations.csv'
# (only for randomisation/blinded/ssz)
# See 'stroke_duplicates.xlsx' for final IDs needs to be removed/corrected
ID_del4 = [2316,1878,2519,2451,664,
           2271,1575,702,2347,1365,
           2024,672,1957,2032,1681,
           394,2001,1996]
stroke = stroke[-stroke["ID"].isin(ID_del4)]  # len(stroke) = 2453
# Manually correct labels for with ID pairs (1681, 1965)
stroke.loc[stroke.ID==1965, 'RandomizationTreatmentControl'] = 0
stroke.loc[stroke.ID==1965, 'BlindedOutcomeAssessment'] = 1
stroke.loc[stroke.ID==1965, 'SampleSizeCalculation'] = 1


# Re-index
stroke.set_index(pd.Series(range(0, len(stroke))), inplace=True)

#%% Output data
# Add columns
stroke['goldID'] = 'stroke' + stroke['ID'].astype(str)  # ID for all the gold data
stroke = stroke.dropna(subset=['RandomizationTreatmentControl', 'BlindedOutcomeAssessment', 'SampleSizeCalculation'], how='all')

stroke.to_csv('rob/rob_stroke_info.txt', sep='\t', encoding='utf-8')
stroke.to_csv('data/stroke/rob_stroke_info.txt', sep='\t', encoding='utf-8')
list(stroke.columns)

#['RecordID',
# 'RandomizationTreatmentControl',
# 'AllocationConcealment',
# 'BlindedOutcomeAssessment',
# 'SampleSizeCalculation',
# 'AnimalExclusions',
# 'Comorbidity',
# 'AnimalWelfareRegulations',
# 'ConflictsOfInterest',
# 'DocumentLink',
# 'fileLink',
# 'ID',
# 'DocumentLink_old',
# 'txtLink',
# 'textLen',
# 'goldID']


#%% Tokenization to json file
stroke = pd.read_csv("data/stroke/rob_stroke_info.txt", sep='\t', engine="python", encoding="utf-8", index_col = 0)   

df = stroke[['goldID',
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


df2json(df_info = df, json_path = 'data/stroke/rob_stroke_fulltokens.json')
