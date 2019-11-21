# -*- coding: utf-8 -*-
"""

Created on Mon Aug 19 10:27:25 2019
@author: qwang

"""

import os
import pandas as pd
import numpy as np


# change working directory
wdir = '/home/qwang/rob/'
os.chdir(wdir)

# from codes.data_process.pdf2text import convert_multiple
from codes.data_process.df2json import df2json
# from codes.data_process.regex import doc_annotate, read_regex
# from codes.data_process.tokenizer import preprocess_text, tokenize_text


#%% Read and format data
psy = pd.read_csv("data/psycho/Psychosis_ROB_categorised.csv", sep=',', engine="python", encoding="utf-8")   
list(psy.columns)
#['RecordID',
# 'PdfRelativePath',
# 'Phase II Categorization\nRandomisation Reported',
# 'Phase II Categorization\nAllocation Concealment Reported',
# 'Phase II Categorization\nBlinded Assessment of Outcome Reported',
# 'Phase II Categorization\nInclusion/Exclusion Criteria Reported',
# 'Phase II Categorization\nSample Size Calculation Reported',
# 'Phase II Categorization\nConflict of Interest Statement Reported',
# 'Phase II Categorization\nCompliance with Animal Welfare Regulations Reported',
# 'Phase II Categorization\nProtocol Availability Reported']

# Change column names
psy.columns = ['RecordID', 'fileLink', 
               
               'RandomizationTreatmentControl',
               'AllocationConcealment',
               'BlindedOutcomeAssessment',
               'AnimalExclusions',
               'SampleSizeCalculation',
               'ConflictsOfInterest',
               'AnimalWelfareRegulations',
               'ProtocolAvailability'] 

psy['ID'] = np.arange(1, len(psy)+1)

# Modify paths
psy['DocumentLink'] = psy['fileLink'].str.replace('S:/JISC Analytics Lab/CAMARADES Datasets/All_psychosis/', 'psycho/PDFs/')


# Manual correction
psy.loc[psy.DocumentLink=='psycho/PDFs/13524_2004.pdf', 'DocumentLink'] = 'psycho/PDFs/13524_2004_updated.pdf'
psy.loc[psy.DocumentLink=='psycho/PDFs/11087_2004.pdf', 'DocumentLink'] = 'psycho/PDFs/11087_2004_updated.pdf'

# Modify pdf path
pdf_folder = '/home/qwang/rob/data/'
psy['fileLink'] = pdf_folder + psy['DocumentLink'].astype(str)

# len(psy) = 2465


#%% Convert pdf to txt by Xpdf
# See pdf2txt_psycho.sh


#%% Check existence of txt files
ID_inv1 = []
link_inv1 = []
for i, row in psy.iterrows():      
    txt_path = 'data/psycho/TXTs/' + os.path.basename(psy.loc[i,'DocumentLink']).split('.')[0] + '.txt'  
    if os.path.exists(txt_path):
        with open(txt_path, 'r', encoding='utf-8') as fp:
            text = fp.read()
        psy.loc[i,'textLen'] = len(text)
    else:
        psy.loc[i,'textLen'] = 0
        ID_inv1.append(psy.loc[i, 'ID'])
        link_inv1.append(psy.loc[i, 'DocumentLink'])
        
# Remove records which do not exist
psy = psy[-psy["ID"].isin(ID_inv1)]  # len(psy) = 2446


#%% Remove records with unique length
psy_dup = psy[psy.duplicated(subset=['textLen'], keep=False)]
psy_dup.loc[:,'textSame'] = ''
# Check whether records with same text length are duplicate
dup_grouped = psy_dup.groupby(['textLen'])
dup_grouped = list(dup_grouped)
len(dup_grouped)  # 70

duplen = []  
for i in range(len(dup_grouped)):
    duplen.append(len(dup_grouped[i][1]))
set(duplen)  # {2, 4, 19}


for i, tup in enumerate(dup_grouped):
    if len(tup[1]) == 2:
        df = tup[1]
        df.set_index(pd.Series(range(0,len(df))), inplace=True)
        path0 = 'data/psycho/TXTs/' + os.path.basename(df['DocumentLink'][0]).split('.')[0] + '.txt'
        path1 = 'data/psycho/TXTs/' + os.path.basename(df['DocumentLink'][1]).split('.')[0] + '.txt'
        with open(path0, 'r', encoding='utf-8') as fp:
            text0 = fp.read()
        with open(path1, 'r', encoding='utf-8') as fp:
            text1 = fp.read()      
        if text0 == text1:
            df['textSame'][0] = 'Yes'

# Convert list to dataframe
frames = [dg[1] for dg in dup_grouped]
dup_df = pd.concat(frames)

dup_df.to_csv('data/psycho/psy_duplicates.csv', sep=',', encoding='utf-8',
              columns = ['ID', 'textSame', 'textLen',                        
                         'RandomizationTreatmentControl',
                         'AllocationConcealment',
                         'BlindedOutcomeAssessment',
                         'AnimalExclusions',
                         'SampleSizeCalculation',
                         'ConflictsOfInterest',
                         'AnimalWelfareRegulations',
                         'ProtocolAvailability',                                   
     
                    'fileLink', 'DocumentLink', 'RecordID'])

# Remove one duplicate record and unextractable records for psychosis data
ID_inv2 = [702,
           125, 681, 669, 2320, 1580,
           2411, 69, 379, 557, 558,
           1052, 2105, 511, 1877, 2040,
           2113, 637, 936, 116, 1902,
           2282, 2370, 889, 2311]
psy = psy[-psy["ID"].isin(ID_inv2)]  # len(psy) = 2421    

# Re
    
#%% Check long and short texts (not from csv file)
print(max(psy['textLen']))  # 512786
print(min(psy['textLen']))  # 6

ID_inv3 = []
for _, row in psy.iterrows():   
    if (row['textLen'] > 100000) or (row['textLen'] < 9000):
        ID_inv3.append(row['ID'])
print('IDs of txts with too long or short texts:\n')
print(ID_inv3)

temp = psy[psy["ID"].isin(ID_inv3)]
temp = temp[['ID', 'textLen', 'DocumentLink']]


# Records with too short/long texts need to be removed (manually checked, see 'psy_issues.xlsx')
ID_inv3 = [1673, 158, 1561, 1618, 958,
           2371, 2215, 1827, 1826, 2056,
           1041, 2055, 1904, 1903, 70,
           578, 1228]
psy = psy[-psy["ID"].isin(ID_inv3)]  # len(psy) = 2404

# Re-index
psy.set_index(pd.Series(range(0, len(psy))), inplace=True)

# Recalculate text length
print(max(psy['textLen']))  # 121447
print(min(psy['textLen']))  # 7444


#%% Output data
# Add columns
for i, row in psy.iterrows():
    psy.loc[i,'txtLink'] = 'data/psycho/TXTs/' + os.path.basename(psy.loc[i,'DocumentLink']).split('.')[0] + '.txt'
psy['goldID'] = 'psy' + psy['ID'].astype(str)  # ID for all the gold data

psy = psy.dropna(subset=['RandomizationTreatmentControl', 'BlindedOutcomeAssessment', 'SampleSizeCalculation'], how='any')

psy.to_csv('data/psycho/rob_psycho_info.txt', sep='\t', encoding='utf-8')
list(psy.columns)

#['RecordID',
# 'fileLink',
# 'RandomizationTreatmentControl',
# 'AllocationConcealment',
# 'BlindedOutcomeAssessment',
# 'AnimalExclusions',
# 'SampleSizeCalculation',
# 'ConflictsOfInterest',
# 'AnimalWelfareRegulations',
# 'ProtocolAvailability',
# 'ID',
# 'DocumentLink',
# 'textLen',
# 'txtLink',
# 'goldID']


#%% Tokenization to json file
psy = pd.read_csv("data/psycho/rob_psycho_info.txt", sep='\t', engine="python", encoding="utf-8", index_col = 0)   

df = psy[['goldID',
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


df2json(df_info = df, json_path = 'data/psycho/rob_psycho_fulltokens.json')






#%% Run regex
# Read data
psy = pd.read_csv("rob_psychosis_fulltext.txt", sep='\t', encoding="utf-8")
df = psy.dropna(subset=['Text'])
df = df[df["Text"]!=' ']
df.set_index(pd.Series(range(0, len(df))), inplace=True)

# Read regex string
reg = 'U:/Datastore/CMVM/scs/groups/DCN/TRIALDEV/CAMARADES/Qianying/RoB/regex/'
regex_randomisation = read_regex(reg+'randomisation.txt')
regex_blinding = read_regex(reg+'blinding.txt')
regex_ssc = read_regex(reg+'ssc.txt')
regex_conflict = read_regex(reg+'conflict.txt')
regex_compliance = read_regex(reg+'compliance.txt')

df['RegexRandomization'] = 0
df['RegexBlinding'] = 0
df['RegexSSC'] = 0
df['RegexConflict'] = 0
df['RegexCompliance'] = 0


# Obtain regex labels 
for i in range(len(df)): 
    df.loc[i,'RegexRandomization'] = doc_annotate(regex_randomisation, df.loc[i,'Text'])
    df.loc[i,'RegexBlinding'] = doc_annotate(regex_blinding, df.loc[i,'Text'])
    df.loc[i,'RegexSSC'] = doc_annotate(regex_ssc, df.loc[i,'Text'])
    df.loc[i,'RegexConflict'] = doc_annotate(regex_conflict, df.loc[i,'Text'])
    df.loc[i,'RegexCompliance'] = doc_annotate(regex_compliance, df.loc[i,'Text']) 
    print(i)


# Compute scores
from sklearn.metrics import confusion_matrix
def compute_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    f1 = 100 * 2*tp / (2*tp+fp+fn)
    accuracy = 100 * (tp+tn) / (tp+tn+fp+fn)
    recall = 100 * tp / (tp+fn)
    specificity = 100 * tn / (tn+fp)
    precision = 100 * tp / (tp+fp)
    print("f1: {0:.2f}% | accuracy: {1:.2f}% | sensitivity: {2:.2f}% | specificity: {3:.2f}% | precision: {4:.2f}%".format(
            f1, accuracy, recall, specificity, precision))

list(df.columns)
compute_score(y_true=df['Randomisation'], y_pred=df['RegexRandomization'])
compute_score(y_true=df['BlindedAssessmentOutcome'], y_pred=df['RegexBlinding'])
compute_score(y_true=df['SampleSizeCalculation'], y_pred=df['RegexSSC'])
compute_score(y_true=df['ConflictInterest'], y_pred=df['RegexConflict'])
compute_score(y_true=df['AnimalWelfare'], y_pred=df['RegexCompliance'])
