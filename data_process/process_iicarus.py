#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 11:53:51 2019

@author: qwang
"""


import os

import pandas as pd
import numpy as np


# Change to src dir
src_dir = '/home/qwang/rob/'
os.chdir(src_dir)
from src.data_process.df2json import df2json

# Change to data dir
data_dir = '/media/mynewdrive/rob/'
os.chdir(data_dir)


#%% Read and format data
review = pd.read_csv('data/iicarus/ReviewFinalResult.csv', sep=',', engine="python", encoding="utf-8")
iicarus = pd.read_csv("data/iicarus/IICARus_RawFinal.csv", sep=',', engine="python", encoding="utf-8")   

# AllocationConcealment
df_conceal = review[(review["CheckListCheckListID"].isin([154])) & (review["Reconcilliation"].isin([True]))][['RecordID', 'OptionName']]
iicarus = pd.merge(iicarus, df_conceal, how='left', on='RecordID')
iicarus.rename(columns={'OptionName':'Conceal'}, inplace=True)
iicarus.Conceal.replace(['Yes', 'No', 'Not Applicable'], [1, 0, 0], inplace=True)

# AnimalWelfareRegulations
df_welfare = review[(review["CheckListCheckListID"].isin([146])) & (review["Reconcilliation"].isin([True]))][['RecordID', 'OptionName']]
iicarus = pd.merge(iicarus, df_welfare, how='left', on='RecordID')
iicarus.rename(columns={'OptionName':'Welfare'}, inplace=True)
iicarus.Welfare.replace(['Yes', 'No', 'Not Applicable'], [1, 0, 0], inplace=True)

# ConflictsOfInterest
df_conflict = review[(review["CheckListCheckListID"].isin([187])) & (review["Reconcilliation"].isin([True]))][['RecordID', 'OptionName']]
iicarus = pd.merge(iicarus, df_conflict, how='left', on='RecordID')
iicarus.rename(columns={'OptionName':'Conflict'}, inplace=True)
iicarus.Conflict.replace(['Yes', 'No', 'Not Applicable'], [1, 0, 0], inplace=True)

# AnimalExclusions
df_exclusion = review[(review["CheckListCheckListID"].isin([168])) & (review["Reconcilliation"].isin([True]))][['RecordID', 'OptionName']]
iicarus = pd.merge(iicarus, df_exclusion, how='left', on='RecordID')
iicarus.rename(columns={'OptionName':'Exclusion'}, inplace=True)
iicarus.Exclusion.replace(['Yes', 'No', 'Not Applicable'], [1, 0, 0], inplace=True)


list(iicarus.columns)
# ['RecordID', 'Randomisation', 'Blinding', 'SampleSize', 'pdfFilePath', 'Conceal', 'Welfare', 'Conflict', 'Exclusion']
# Chancge column names
iicarus.rename(columns={'Randomisation':'RandomizationTreatmentControl',
                        'Blinding': 'BlindedOutcomeAssessment',
                        'SampleSize': 'SampleSizeCalculation',
                        'Conceal': 'AllocationConcealment',
                        'Welfare': 'AnimalWelfareRegulations',
                        'Conflict': 'ConflictsOfInterest',
                        'Exclusion': 'AnimalExclusions'}, inplace=True)

iicarus['ID'] = np.arange(1, len(iicarus)+1)


# Modify paths
iicarus['pdfFilePath'] = iicarus['pdfFilePath'].str.replace('\\', '/')

iicarus['DocumentLink'] = iicarus['pdfFilePath']
iicarus['fileLink'] = iicarus['pdfFilePath']
iicarus['txtLink'] = iicarus['pdfFilePath']

for i, row in iicarus.iterrows():
    url = iicarus.loc[i,'pdfFilePath']
    url = url.split(";")[0]
    if len(os.path.basename(url).split('.pdf')) == 1:
        url = url + '.pdf'
    iicarus.loc[i,'fileLink'] = url
    iicarus.loc[i,'DocumentLink'] = os.path.join('data/iicarus/PDFs', os.path.basename(url))
    
# Manual correction
# Replace old/blinded PDFs
iicarus.loc[iicarus.DocumentLink=='data/iicarus/PDFs/PONE-D-15-21944stripped.pdf', 'DocumentLink'] = 'data/iicarus/PDFs/manual/PONE-D-15-21944stripped_new.pdf'
iicarus.loc[iicarus.DocumentLink=='data/iicarus/PDFs/PONE-D-15-19338stripped.pdf', 'DocumentLink'] = 'data/iicarus/PDFs/manual/PONE-D-15-19338stripped_new.pdf'
iicarus.loc[iicarus.DocumentLink=='data/iicarus/PDFs/PONE-D-15-14915stripped.pdf', 'DocumentLink'] = 'data/iicarus/PDFs/manual/PONE-D-15-14915stripped_new.pdf'
iicarus.loc[iicarus.DocumentLink=='data/iicarus/PDFs/PONE-D-15-15072stripped.pdf', 'DocumentLink'] = 'data/iicarus/PDFs/manual/PONE-D-15-15072stripped_new.pdf'
iicarus.loc[iicarus.DocumentLink=='data/iicarus/PDFs/PONE-D-15-11276stripped.pdf', 'DocumentLink'] = 'data/iicarus/PDFs/manual/PONE-D-15-11276stripped_new.pdf'

iicarus.loc[iicarus.DocumentLink=='data/iicarus/PDFs/PONE-D-15-19862stripped.pdf', 'DocumentLink'] = 'data/iicarus/PDFs/manual/PONE-D-15-19862stripped_new.pdf'
iicarus.loc[iicarus.DocumentLink=='data/iicarus/PDFs/PONE-D-15-17526stripped.pdf', 'DocumentLink'] = 'data/iicarus/PDFs/manual/PONE-D-15-17526stripped_new.pdf'
iicarus.loc[iicarus.DocumentLink=='data/iicarus/PDFs/PONE-D-15-13863stripped.pdf', 'DocumentLink'] = 'data/iicarus/PDFs/manual/PONE-D-15-13863stripped_new.pdf'
iicarus.loc[iicarus.DocumentLink=='data/iicarus/PDFs/PONE-D-15-18907stripped.pdf', 'DocumentLink'] = 'data/iicarus/PDFs/manual/PONE-D-15-18907stripped_new.pdf'
iicarus.loc[iicarus.DocumentLink=='data/iicarus/PDFs/PONE-D-15-18783stripped.pdf', 'DocumentLink'] = 'data/iicarus/PDFs/manual/PONE-D-15-18783stripped_new.pdf'

iicarus.loc[iicarus.DocumentLink=='data/iicarus/PDFs/PONE-D-15-14015stripped.pdf', 'DocumentLink'] = 'data/iicarus/PDFs/manual/PONE-D-15-14015stripped_new.pdf'
iicarus.loc[iicarus.DocumentLink=='data/iicarus/PDFs/PONE-D-15-20694stripped.pdf', 'DocumentLink'] = 'data/iicarus/PDFs/manual/PONE-D-15-20694stripped_new.pdf'
iicarus.loc[iicarus.DocumentLink=='data/iicarus/PDFs/PONE-D-15-19789stripped.pdf', 'DocumentLink'] = 'data/iicarus/PDFs/manual/PONE-D-15-19789stripped_new.pdf'
iicarus.loc[iicarus.DocumentLink=='data/iicarus/PDFs/PONE-D-15-20478stripped.pdf', 'DocumentLink'] = 'data/iicarus/PDFs/manual/PONE-D-15-20478stripped_new.pdf'
iicarus.loc[iicarus.DocumentLink=='data/iicarus/PDFs/PONE-D-15-17397stripped.pdf', 'DocumentLink'] = 'data/iicarus/PDFs/manual/PONE-D-15-17397stripped_new.pdf'

iicarus.loc[iicarus.DocumentLink=='data/iicarus/PDFs/PONE-D-15-20282stripped.pdf', 'DocumentLink'] = 'data/iicarus/PDFs/manual/PONE-D-15-20282stripped_new.pdf'
iicarus.loc[iicarus.DocumentLink=='data/iicarus/PDFs/PONE-D-15-22112stripped.pdf', 'DocumentLink'] = 'data/iicarus/PDFs/manual/PONE-D-15-22112stripped_new.pdf'
iicarus.loc[iicarus.DocumentLink=='data/iicarus/PDFs/PONE-D-15-17347stripped.pdf', 'DocumentLink'] = 'data/iicarus/PDFs/manual/PONE-D-15-17347stripped_new.pdf'
iicarus.loc[iicarus.DocumentLink=='data/iicarus/PDFs/PONE-D-15-19867stripped.pdf', 'DocumentLink'] = 'data/iicarus/PDFs/manual/PONE-D-15-19867stripped_new.pdf'
iicarus.loc[iicarus.DocumentLink=='data/iicarus/PDFs/PONE-D-15-15322stripped.pdf', 'DocumentLink'] = 'data/iicarus/PDFs/manual/PONE-D-15-15322stripped_new.pdf'

iicarus.loc[iicarus.DocumentLink=='data/iicarus/PDFs/PONE-D-15-15847stripped.pdf', 'DocumentLink'] = 'data/iicarus/PDFs/manual/PONE-D-15-15847stripped_new.pdf'
iicarus.loc[iicarus.DocumentLink=='data/iicarus/PDFs/PONE-D-15-17809stripped.pdf', 'DocumentLink'] = 'data/iicarus/PDFs/manual/PONE-D-15-17809stripped_new.pdf'
iicarus.loc[iicarus.DocumentLink=='data/iicarus/PDFs/PONE-D-15-18465stripped.pdf', 'DocumentLink'] = 'data/iicarus/PDFs/manual/PONE-D-15-18465stripped_new.pdf'
iicarus.loc[iicarus.DocumentLink=='data/iicarus/PDFs/PONE-D-15-16235stripped.pdf', 'DocumentLink'] = 'data/iicarus/PDFs/manual/PONE-D-15-16235stripped_new.pdf'
iicarus.loc[iicarus.DocumentLink=='data/iicarus/PDFs/PONE-D-15-20471stripped.pdf', 'DocumentLink'] = 'data/iicarus/PDFs/manual/PONE-D-15-20471stripped_new.pdf'

iicarus.loc[iicarus.DocumentLink=='data/iicarus/PDFs/PONE-D-15-18075stripped.pdf', 'DocumentLink'] = 'data/iicarus/PDFs/manual/PONE-D-15-18075stripped_new.pdf'
iicarus.loc[iicarus.DocumentLink=='data/iicarus/PDFs/PONE-D-15-18058stripped.pdf', 'DocumentLink'] = 'data/iicarus/PDFs/manual/PONE-D-15-18058stripped_new.pdf'
iicarus.loc[iicarus.DocumentLink=='data/iicarus/PDFs/PONE-D-15-11791stripped.pdf', 'DocumentLink'] = 'data/iicarus/PDFs/manual/PONE-D-15-11791stripped_new.pdf'
#iicarus.loc[iicarus.DocumentLink=='data/iicarus/PDFs/.pdf', 'DocumentLink'] = 'data/iicarus/PDFs/manual/.pdf'
#iicarus.loc[iicarus.DocumentLink=='data/iicarus/PDFs/.pdf', 'DocumentLink'] = 'data/iicarus/PDFs/manual/.pdf'

# Modify txt paths
iicarus['txtLink'] = iicarus['DocumentLink'].str.replace('.pdf', ".txt") 
iicarus['txtLink'] = iicarus['txtLink'].str.replace('PDFs/manual', "TXTs")
iicarus['txtLink'] = iicarus['txtLink'].str.replace('PDFs', "TXTs")



#%% Convert pdf to txt by Xpdf
# See pdf2txt_iicarus.sh

#%% Check existence of txt files
ID_del1 = []
link_del1 = []
for i, row in iicarus.iterrows():      
    txt_path = iicarus.loc[i,'txtLink']  
    if os.path.exists(txt_path):
        with open(txt_path, 'r', encoding='utf-8') as fp:
            text = fp.read()
        iicarus.loc[i,'textLen'] = len(text)
    else:
        iicarus.loc[i,'textLen'] = 0
        
    if iicarus.loc[i,'textLen'] < 1000:
        ID_del1.append(iicarus.loc[i, 'ID'])
        link_del1.append(iicarus.loc[i, 'DocumentLink'])

      
# Remove records which do not exist or too short
iicarus = iicarus[-iicarus["ID"].isin(ID_del1)]  # len(npqip) = 665


#%% Check long and short texts (not from csv file)
print(max(iicarus['textLen']))  # 123806
print(min(iicarus['textLen']))  # 15603

ID_del2 = []
for _, row in iicarus.iterrows():   
    if (row['textLen'] > 100000) or (row['textLen'] < 9000):
        ID_del2.append(row['ID'])
print('IDs of txts with too long or short texts:\n')
print(ID_del2)

temp = iicarus[iicarus["ID"].isin(ID_del2)]
temp = temp[['ID', 'textLen', 'DocumentLink']]


# Records with too short/long texts need to be removed (manually checked, see 'iicarus_issues.xlsx')

# Re-index
iicarus.set_index(pd.Series(range(0, len(iicarus))), inplace=True)

#%% Remove records with unique length
iicarus_dup = iicarus[iicarus.duplicated(subset=['textLen'], keep=False)]
iicarus_dup.loc[:,'textSame'] = ''
# Check whether records with same text length are duplicate
dup_grouped = iicarus_dup.groupby(['textLen'])
dup_grouped = list(dup_grouped)
len(dup_grouped)  # 3

duplen = []  
for i in range(len(dup_grouped)):
    duplen.append(len(dup_grouped[i][1]))
set(duplen)  # {2}

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

# dup_df.to_csv('data/npqip/iicarus_duplicates.csv', sep=',', encoding='utf-8')
# Remove one duplicate records (No duplicates)

#%% Output data
# Add columns
iicarus['goldID'] = 'iicarus' + iicarus['ID'].astype(str)  # ID for all the gold data

iicarus['RandomizationTreatmentControl'].fillna(0, inplace=True)
iicarus['BlindedOutcomeAssessment'].fillna(0, inplace=True)
iicarus['SampleSizeCalculation'].fillna(0, inplace=True)
iicarus['AllocationConcealment'].fillna(0, inplace=True)
iicarus['AnimalWelfareRegulations'].fillna(0, inplace=True)
iicarus['ConflictsOfInterest'].fillna(0, inplace=True)
iicarus['AnimalExclusions'].fillna(0, inplace=True)

# For Jing to check
#missing = iicarus[pd.isnull(iicarus['RandomizationTreatmentControl']) | pd.isnull(iicarus['BlindedOutcomeAssessment']) | pd.isnull(iicarus['SampleSizeCalculation'])]
#missing.to_csv('data/iicarus/iicarus_missing.csv', sep=',', encoding='utf-8')    

# Drop records if any of the rob items has missing values
iicarus = iicarus.dropna(subset=['RandomizationTreatmentControl', 'BlindedOutcomeAssessment', 'SampleSizeCalculation', 
                                 'AllocationConcealment', 'AnimalWelfareRegulations', 'ConflictsOfInterest', 'AnimalExclusions'], how='any')


# Type conversion
iicarus.RandomizationTreatmentControl = iicarus.RandomizationTreatmentControl.astype(int)
iicarus.BlindedOutcomeAssessment = iicarus.BlindedOutcomeAssessment.astype(int)
iicarus.SampleSizeCalculation = iicarus.SampleSizeCalculation.astype(int)
iicarus.AllocationConcealment = iicarus.AllocationConcealment.astype(int)
iicarus.AnimalWelfareRegulations = iicarus.AnimalWelfareRegulations.astype(int)
iicarus.ConflictsOfInterest = iicarus.ConflictsOfInterest.astype(int)
iicarus.AnimalExclusions = iicarus.AnimalExclusions.astype(int)


iicarus.to_csv('data/iicarus/rob_iicarus_info.txt', sep='\t', encoding='utf-8')
list(iicarus.columns)


#    ['RecordID',
#     'RandomizationTreatmentControl',
#     'BlindedOutcomeAssessment',
#     'SampleSizeCalculation',
#     'pdfFilePath',
#     'AllocationConcealment',
#     'AnimalWelfareRegulations',
#     'ConflictsOfInterest',
#     'AnimalExclusions',
#     'ID',
#     'DocumentLink',
#     'fileLink',
#     'txtLink',
#     'textLen',
#     'goldID']



#%% Tokenization to json file
#iicarus = pd.read_csv("data/iicarus/rob_iicarus_info.txt", sep='\t', engine="python", encoding="utf-8", index_col = 0)   
#iicarus['AllocationConcealment'] = float('nan')
#iicarus['AnimalWelfareRegulations'] = float('nan')
#iicarus['ConflictsOfInterest'] = float('nan')
#iicarus['AnimalExclusions'] = float('nan')
#
#df = iicarus[['goldID',
#              'fileLink',
#              'DocumentLink',
#              'txtLink',
#              'RandomizationTreatmentControl',
#              'AllocationConcealment',
#              'BlindedOutcomeAssessment',
#              'SampleSizeCalculation',
#              'AnimalWelfareRegulations',
#              'ConflictsOfInterest',
#              'AnimalExclusions']]
#
#
#df2json(df_info = df, json_path = 'data/iicarus/rob_iicarus_fulltokens.json')
#
#
##%% Tokenization to json file (Grobid)
#iicarus = pd.read_csv("data/iicarus/rob_iicarus_info.txt", sep='\t', engine="python", encoding="utf-8", index_col = 0)   
#iicarus['txtLink'] = iicarus['txtLink'].str.replace('TXTs', "GROTXTs")
#iicarus['txtLink'] = iicarus['txtLink'].str.replace('.txt', ".tei.txt")
##'data/np/TXTs/np_1.txt'
##'data/np/GROTXTs/np_1.tei.txt'
#
#iicarus['AllocationConcealment'] = float('nan')
#iicarus['AnimalWelfareRegulations'] = float('nan')
#iicarus['ConflictsOfInterest'] = float('nan')
#iicarus['AnimalExclusions'] = float('nan')
#
#
#iicarus.to_csv("data/iicarus/rob_iicarus_info_grobid.txt", sep='\t', encoding="utf-8")   
#
#
#df = iicarus[['goldID',
#            'fileLink',
#            'DocumentLink',
#            'txtLink',
#            'RandomizationTreatmentControl',
#            'AllocationConcealment',
#            'BlindedOutcomeAssessment',
#            'SampleSizeCalculation',
#            'AnimalWelfareRegulations',
#            'ConflictsOfInterest',
#            'AnimalExclusions']]
#
#df2json(df_info = df, json_path = 'data/iicarus/rob_iicarus_fulltokens_grobid.json')
