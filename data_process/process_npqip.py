#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 21:31:33 2019

@author: qwang
"""

import os

import pandas as pd
import numpy as np

# change working directory
wdir = '/home/qwang/rob/'
os.chdir(wdir)

from src.data_process.df2json import df2json


#%% Read and format data
npqip = pd.read_csv("data/npqip/NPQIP_RawFinal.csv", sep=',', engine="python", encoding="utf-8")   
list(npqip.columns)
# ['PublicationNumber', 'Randomisation', 'Blinding', 'SampleSize', 'pdfFilePath']
npqip.columns = ['PublicationNumber', 
               'RandomizationTreatmentControl',
               'BlindedOutcomeAssessment',
               'SampleSizeCalculation',
               'pdfFilePath'] 


npqip['ID'] = np.arange(1, len(npqip)+1)
# Chancge column names



# Modify paths
npqip['pdfFilePath'] = npqip['pdfFilePath'].str.replace('\\', '/')


npqip['DocumentLink'] = npqip['pdfFilePath']
npqip['fileLink'] = npqip['pdfFilePath']
npqip['txtLink'] = npqip['pdfFilePath']

for i, row in npqip.iterrows():
    url = npqip.loc[i,'pdfFilePath']
    url = url.split(";")[0]
    if len(os.path.basename(url).split('.pdf')) == 1:
        url = url + '.pdf'
    npqip.loc[i,'fileLink'] = url
    npqip.loc[i,'DocumentLink'] = os.path.join('data/npqip/PDFs', os.path.basename(url))
    
# Manual correction
# Replace old/blinded PDFs
npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NSS104_pair_redact.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NSS104_pair_redact_new.pdf'
npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NPG312.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NPG312_new.pdf'
npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/new_NSS245_pair_Redacted.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/new_NSS245_pair_Redacted_new.pdf'
npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NPG299.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NPG299_new.pdf'
npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NPG210.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NPG210_new.pdf'

npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NPG317.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NPG317_new.pdf'
npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NSS241_pair_Redacted.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NSS241_pair_Redacted_new.pdf'
npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NPG236.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NPG236_new.pdf'
npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NSS422_pair_Redacted.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NSS422_pair_Redacted_new.pdf'
npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NPG362.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NPG362_new.pdf'

npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NPG379.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NPG379_new.pdf'
npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NPG262.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NPG262_new.pdf'
npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NSS246_pair_Redacted.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NSS246_pair_Redacted_new.pdf'
npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NSS421_pair_Redacted.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NSS421_pair_Redacted_new.pdf'
npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/new_NSS363_pair_Redacted.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/new_NSS363_pair_Redacted_new.pdf'

npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NPG439.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NPG439_new.pdf'
npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NSS085_pair_Redacted.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NSS085_pair_Redacted_new.pdf'
npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NSS263_pair_Redacted.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NSS263_pair_Redacted_new.pdf'
npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NPG007.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NPG007_new.pdf'
npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NSS094_pair_Redact.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NSS094_pair_Redact_new.pdf'

npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NPG308.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NPG308_new.pdf'
npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NPG337.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NPG337_new.pdf'
npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NSS434_pair_Redacted.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NSS434_pair_Redacted_new.pdf'
npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NSS160_pair_Redact.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NSS160_pair_Redact_new.pdf'
npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NSS289_pair_Redacted.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NSS289_pair_Redacted_new.pdf'

npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NPG241.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NPG241_new.pdf'
npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NPG322.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NPG322_new.pdf'
npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NSS407_pair_Redacted.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NSS407_pair_Redacted_new.pdf'
npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NSS253_pair_Redacted.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NSS253_pair_Redacted_new.pdf'
npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NPG249.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NPG249_new.pdf'

npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/nss126_pair_Redact.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/nss126_pair_Redact_new.pdf'
npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NSS153_pair_Redact.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NSS153_pair_Redact_new.pdf'
npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NSS262_pair_Redacted.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NSS262_pair_Redacted_new.pdf'
npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NSS147_pair_Redacted.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NSS147_pair_Redacted_new.pdf'
npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NSS070_pair_Redact.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NSS070_pair_Redact_new.pdf'

npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NPG170.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NPG170_new.pdf'

# Replace PDFs containing supplementary data
npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NPG080.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NPG080_new.pdf'
npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NPG381.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NPG381_new.pdf'
npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NPG099.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NPG099_new.pdf'
npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NPG387.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NPG387_new.pdf'
npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NPG284.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NPG284_new.pdf'

npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NPG109.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NPG109_new.pdf'
npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NPG147.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NPG147_new.pdf'
npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NPG102.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NPG102_new.pdf'
npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NPG292.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NPG292_new.pdf'
npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NPG427.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NPG427_new.pdf'

npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NPG110.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NPG110_new.pdf'
npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NPG168.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NPG168_new.pdf'
npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NPG392.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NPG392_new.pdf'
npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NPG374.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NPG374_new.pdf'
npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NPG334.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NPG334_new.pdf'

npqip.loc[npqip.DocumentLink=='data/npqip/PDFs/NSS282_pair_Redacted.pdf', 'DocumentLink'] = 'data/npqip/PDFs/manual/NSS282_pair_Redacted_new.pdf'


# Modify txt paths
npqip['txtLink'] = npqip['DocumentLink'].str.replace('.pdf', ".txt") 
npqip['txtLink'] = npqip['txtLink'].str.replace('PDFs/manual', "TXTs")
npqip['txtLink'] = npqip['txtLink'].str.replace('PDFs', "TXTs")
       


#%% Convert pdf to txt by Xpdf
# See pdf2txt_npqip.sh


#%% Check existence of txt files
ID_del1 = []
link_del1 = []
for i, row in npqip.iterrows():      
    txt_path = npqip.loc[i,'txtLink']  
    if os.path.exists(txt_path):
        with open(txt_path, 'r', encoding='utf-8') as fp:
            text = fp.read()
        npqip.loc[i,'textLen'] = len(text)
    else:
        npqip.loc[i,'textLen'] = 0
        
    if npqip.loc[i,'textLen'] < 1000:
        ID_del1.append(npqip.loc[i, 'ID'])
        link_del1.append(npqip.loc[i, 'DocumentLink'])

      
# Remove records which do not exist or too short
npqip = npqip[-npqip["ID"].isin(ID_del1)]  # len(npqip) = 762


#%% Check long and short texts (not from csv file)
print(max(npqip['textLen']))  # 108469
print(min(npqip['textLen']))  # 9553

ID_del2 = []
for _, row in npqip.iterrows():   
    if (row['textLen'] > 100000) or (row['textLen'] < 9000):
        ID_del2.append(row['ID'])
print('IDs of txts with too long or short texts:\n')
print(ID_del2)

temp = npqip[npqip["ID"].isin(ID_del2)]
temp = temp[['ID', 'textLen', 'DocumentLink']]


# Records with too short/long texts need to be removed (manually checked, see 'npqip_issues.xlsx')

# Re-index
npqip.set_index(pd.Series(range(0, len(npqip))), inplace=True)



#%% Remove records with unique length
npqip_dup = npqip[npqip.duplicated(subset=['textLen'], keep=False)]
npqip_dup.loc[:,'textSame'] = ''
# Check whether records with same text length are duplicate
dup_grouped = npqip_dup.groupby(['textLen'])
dup_grouped = list(dup_grouped)
len(dup_grouped)  # 9

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

dup_df.to_csv('data/npqip/npqip_duplicates.csv', sep=',', encoding='utf-8')

# Remove one duplicate records
ID_del3 = [825, 878]
npqip = npqip[-npqip["ID"].isin(ID_del3)]  # len(npqip) = 760    

# Re-index
npqip.set_index(pd.Series(range(0, len(npqip))), inplace=True)


#%% Output data
# Add columns
npqip['goldID'] = 'npqip' + npqip['ID'].astype(str)  # ID for all the gold data
npqip = npqip.dropna(subset=['RandomizationTreatmentControl', 'BlindedOutcomeAssessment', 'SampleSizeCalculation'], how='any')

# Type conversion
npqip.RandomizationTreatmentControl = npqip.RandomizationTreatmentControl.astype(int)
npqip.BlindedOutcomeAssessment = npqip.BlindedOutcomeAssessment.astype(int)
npqip.SampleSizeCalculation = npqip.SampleSizeCalculation.astype(int)

npqip.to_csv('data/npqip/rob_npqip_info.txt', sep='\t', encoding='utf-8')
list(npqip.columns)


#['PublicationNumber',
# 'RandomizationTreatmentControl',
# 'BlindedOutcomeAssessment',
# 'SampleSizeCalculation',
# 'pdfFilePath',
# 'ID',
# 'DocumentLink',
# 'fileLink',
# 'txtLink',
# 'textLen',
# 'goldID']


#%% Tokenization to json file
npqip = pd.read_csv("data/npqip/rob_npqip_info.txt", sep='\t', engine="python", encoding="utf-8", index_col = 0)   
npqip['AllocationConcealment'] = float('nan')
npqip['AnimalWelfareRegulations'] = float('nan')
npqip['ConflictsOfInterest'] = float('nan')
npqip['AnimalExclusions'] = float('nan')

df = npqip[['goldID',
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


df2json(df_info = df, json_path = 'data/npqip/rob_npqip_fulltokens.json')


