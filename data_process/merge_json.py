#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 18:05:06 2019

@author: qwang
"""

import os
import shutil
import json
import pandas as pd
import random
import re
import pickle
from tqdm import tqdm
random.seed(1234)

os.chdir('/home/qwang/rob/rob-pome')
from data_process.regex import doc_annotate, read_regex
os.chdir('/home/qwang/rob')
regex_random = read_regex('regex/RandomizationTreatmentControl.txt')
regex_blind = read_regex('regex/BlindedOutcomeAssessment.txt')
regex_sample = read_regex('regex/SampleSizeCalculation.txt')
regex_conflict = read_regex('regex/ConflictsOfInterest.txt')
regex_welfare = read_regex('regex/AnimalWelfareRegulations.txt')

os.chdir('/media/mynewdrive/rob/data')


#%% Merge json files
def read_json(json_path):
    df = []
    with open(json_path, 'r') as fin:
        for line in fin:
            df.append(json.loads(line))
    return df

stroke = read_json(json_path='stroke/rob_stroke_fulltokens.json')
psy = read_json(json_path='psycho/rob_psycho_fulltokens.json')
neuro = read_json(json_path='np/rob_np_fulltokens.json')
npqip = read_json(json_path='npqip/rob_npqip_fulltokens.json')
iicarus = read_json(json_path='iicarus/rob_iicarus_fulltokens.json')

stroke_gro = read_json(json_path='stroke/rob_stroke_fulltokens_grobid.json')
psy_gro = read_json(json_path='psycho/rob_psycho_fulltokens_grobid.json')
neuro_gro = read_json(json_path='np/rob_np_fulltokens_grobid.json')
npqip_gro = read_json(json_path='npqip/rob_npqip_fulltokens_grobid.json')
iicarus_gro = read_json(json_path='iicarus/rob_iicarus_fulltokens_grobid.json')


# Merge data
gold = stroke + psy + neuro + npqip + iicarus  # len(gold)=7908
gold_gro = stroke_gro + psy_gro + neuro_gro + npqip_gro + iicarus_gro

# Grobid



#%% Check token length
# Remove records with too short token length
goldID_del = []
num_tokens = []

for g in gold:
    if len(g['textTokens']) < 2000:
        goldID_del.append(g['goldID'])
    else:
        num_tokens.append(len(g['textTokens']))
        
gold_final = [g for g in gold if g['goldID'] not in goldID_del]  # 7877

print(max(num_tokens))  # 32289
print(min(num_tokens))  # 2002


### For grobid ### 
goldID_del = []
num_tokens = []

for g in gold_gro:
    if len(g['textTokens']) < 1000:
        goldID_del.append(g['goldID'])
    else:
        num_tokens.append(len(g['textTokens']))


print(len(goldID_del))  # 188 (<1000); 547 (<2000)..
        
gold_gro_final = [g for g in gold_gro if g['goldID'] not in goldID_del]  # 7877

print(max(num_tokens))  # 17999
print(min(num_tokens))  # 1012


# Histogram for number of tokens
import matplotlib.pyplot as plt
plt.hist(num_tokens, bins=40, edgecolor='black', alpha=0.8)
plt.xlabel("Number of tokens")
plt.ylabel("Frequency")
plt.show()

#%% Output
random.shuffle(gold_final)
with open('rob_gold_tokens.json', 'w') as fout:
    for dic in gold_final:     
        fout.write(json.dumps(dic) + '\n')   
    
       
### Generate index file                
gold_df = pd.DataFrame(columns = ['goldID', 'fileLink', 'DocumentLink', 'txtLink',
                                  'RandomizationTreatmentControl', 'AllocationConcealment', 'BlindedOutcomeAssessment', 
                                  'SampleSizeCalculation', 'AnimalWelfareRegulations', 'ConflictsOfInterest', 'AnimalExclusions'])

for g in gold_final:
    gold_df = gold_df.append({'goldID': g['goldID'], 'fileLink': g['fileLink'], 'DocumentLink': g['DocumentLink'], 'txtLink': g['txtLink'],
                            'RandomizationTreatmentControl': g['RandomizationTreatmentControl'],
                            'AllocationConcealment': g['AllocationConcealment'],
                            'BlindedOutcomeAssessment': g['BlindedOutcomeAssessment'],
                            'SampleSizeCalculation': g['SampleSizeCalculation'],
                            'AnimalWelfareRegulations': g['AnimalWelfareRegulations'],
                            'ConflictsOfInterest': g['ConflictsOfInterest'],
                            'AnimalExclusions': g['AnimalExclusions']
                            }, ignore_index=True)
           
        
gold_df.to_csv('rob_gold_info.txt', sep='\t', encoding='utf-8')
        
   
#%% Error correction - 1
# Generate error checking records for random/blinded/samplesize (randomly 5% from each project)
g_stroke = [g for g in gold if re.sub(r'[0-9]', '', g['goldID']) == 'stroke'] 
g_np = [g for g in gold if re.sub(r'[0-9]', '', g['goldID']) == 'np'] 
g_psy = [g for g in gold if re.sub(r'[0-9]', '', g['goldID']) == 'psy'] 
g_npqip = [g for g in gold if re.sub(r'[0-9]', '', g['goldID']) == 'npqip'] 
g_iicarus = [g for g in gold if re.sub(r'[0-9]', '', g['goldID']) == 'iicarus'] 

random.shuffle(g_stroke)
random.shuffle(g_np)
random.shuffle(g_psy)
random.shuffle(g_npqip)
random.shuffle(g_iicarus)

gold_err = g_stroke[:int(len(g_stroke)*0.05)] + \
           g_np[:int(len(g_np)*0.05)] + \
           g_psy[:int(len(g_psy)*0.05)] + \
           g_npqip[:int(len(g_npqip)*0.05)] + \
           g_iicarus[:int(len(g_iicarus)*0.05)]
           
           
# Create data frame for SyRF          
err_df = pd.DataFrame(columns = ['Title', 'Authors', 'Publication Name', 'Alternate Name', 'Abstract',
                                 'Url', 'Author Address', 'Year', 'DOI', 
                                 'Keywords', 'Reference Type', 'PDF Relative Path'])

for g in gold_err:
    err_df = err_df.append({'Title': "", 'Authors': "", 'Publication Name': "", 'Alternate Name': "", 'Abstract': "",
                            'Url': "", 'Author Address': "", 'Year': "", 'DOI': "", 
                            'Keywords': g['goldID'], 'Reference Type': g['goldID'], 'PDF Relative Path': g['DocumentLink'],
                            'RandomizationTreatmentControl': g['RandomizationTreatmentControl'],
                            'AllocationConcealment': g['AllocationConcealment'],
                            'BlindedOutcomeAssessment': g['BlindedOutcomeAssessment'],
                            'SampleSizeCalculation': g['SampleSizeCalculation'],
                            'AnimalWelfareRegulations': g['AnimalWelfareRegulations'],
                            'ConflictsOfInterest': g['ConflictsOfInterest'],
                            'AnimalExclusions': g['AnimalExclusions']
                            }, ignore_index=True)
   
   

# Modify Reference type
err_df['Reference Type'] = err_df['Reference Type'].str.replace(r'[0-9]', '')  

# Copy PDFs to 'error_review' folder
err_df['PDF Relative Path'] = err_df['PDF Relative Path'].str.replace('psycho/', "data/psycho/")

os.chdir('/media/mynewdrive/rob')
for i, row in err_df.iterrows():  
    old_path = err_df.loc[i,'PDF Relative Path']            
    new_path = os.path.join('data/error_review', err_df.loc[i,'Reference Type'] , os.path.basename(old_path))
    err_df.loc[i,'PDF Relative Path'] = new_path    
    if os.path.exists(old_path) == False:
        print('PDF does not exist: {}'.format(old_path))
    else:        
        shutil.copy2(old_path, new_path)


err_df.to_csv('data/error_review/rob_error_checking.txt', sep='\t', encoding='utf-8')



#%% Error correction - 2
# Compare annotations for error_review (from MM)
err = pd.read_csv("error_review/rob_error_checking.txt", sep='\t', engine="python", encoding="utf-8")   
err = err[['Unnamed: 0', 'Keywords', 'RandomizationTreatmentControl', 'BlindedOutcomeAssessment', 'SampleSizeCalculation', 'PDF Relative Path']]
err.rename(columns={'Unnamed: 0': 'ID', 'Keywords': 'goldID'}, inplace=True)


mal = pd.read_csv("error_review/rob_annotation_malcolm.csv", sep=',', engine="python", encoding="utf-8")
mal = mal[['Title', 'Randomisation', 'Blinding', 'SampleSizeCalculation', 'StudyId']]
mal.rename(columns={'Title': 'ID', 'Randomisation': 'RandomizationTreatmentControl', 'Blinding': 'BlindedOutcomeAssessment'}, inplace=True)
mal = mal.replace({True: 1, False: 0})

res = pd.merge(mal, err, how='left', on='ID')

for i in range(len(res)):
    res.loc[i, 'group'] = re.sub(r'\d+', "", res.loc[i, 'goldID'])


df = res[res.group == 'psy']
df.set_index(pd.Series(range(0, len(df))), inplace=True)
tr = df[df.RandomizationTreatmentControl_y == df.RandomizationTreatmentControl_x]
tb = df[df.BlindedOutcomeAssessment_y == df.BlindedOutcomeAssessment_x]
ts = df[df.SampleSizeCalculation_y == df.SampleSizeCalculation_x]
print(round(len(tr)/len(df)*100,2), round(len(tb)/len(df)*100,2), round(len(ts)/len(df)*100,2))

# Output inconsistent records between gold data and Malcolm's annotation
df = res
df_mis = df[(df.RandomizationTreatmentControl_y != df.RandomizationTreatmentControl_x) | 
            (df.BlindedOutcomeAssessment_y != df.BlindedOutcomeAssessment_x) |
            (df.SampleSizeCalculation_y != df.SampleSizeCalculation_x)]
df_mis.rename(columns={'RandomizationTreatmentControl_x': 'Randomization_M', 
                       'BlindedOutcomeAssessment_x': 'Blinded_M', 
                       'SampleSizeCalculation_x': 'SSC_M',
                       'RandomizationTreatmentControl_y': 'Randomization_G', 
                       'BlindedOutcomeAssessment_y': 'Blinded_G', 
                       'SampleSizeCalculation_y': 'SSC_G'                     
                       }, inplace=True)

df_mis.to_csv('error_review/rob_error_inconsistent.csv', sep=',', encoding='utf-8')


# Generate error checking records for exclusion/welfare/conceal/conflict
df_info = pd.read_pickle('rob_info_a.pkl') 
df_info.rename(columns={'goldID': 'Keywords'}, inplace=True)
df_err = pd.read_csv('error_review/rob_error_checking.txt', sep='\t', encoding='utf-8')
df_err.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)

res = pd.merge(df_err, df_info, how='left', on='Keywords')
res = res[[  'ID',
             'Title',
             'Authors',
             'Publication Name',
             'Alternate Name',
             'Abstract',
             'Url',
             'Author Address',
             'Year',
             'DOI',
             'Keywords',
             'Reference Type',
             'PDF Relative Path',
             'RandomizationTreatmentControl_y',
             'AllocationConcealment_y',
             'BlindedOutcomeAssessment_y',
             'SampleSizeCalculation_y',
             'AnimalWelfareRegulations_y',
             'ConflictsOfInterest_y',
             'AnimalExclusions_y',
             'partition']]

res.rename(columns={'RandomizationTreatmentControl_y': 'RandomizationTreatmentControl', 
                    'BlindedOutcomeAssessment_y': 'BlindedOutcomeAssessment', 
                    'SampleSizeCalculation_y': 'SampleSizeCalculation',
                    'AllocationConcealment_y': 'AllocationConcealment', 
                    'AnimalWelfareRegulations_y': 'AnimalWelfareRegulations', 
                    'ConflictsOfInterest_y': 'ConflictsOfInterest',
                    'AnimalExclusions_y': 'AnimalExclusions'                      
                    }, inplace=True)

res.to_csv('error_review/rob_error_check_update.txt', sep='\t', encoding='utf-8')

#%% Error correction - 3 
## Error checking for gold data (from regex)
err = pd.read_csv("error_review/rob_error_checking.txt", sep='\t', engine="python", encoding="utf-8")  
id_checked = list(err['Keywords'])  #ids of records checked by MM

# Get regex label
df = pd.read_pickle('rob_info_a.pkl')  # for random/blind/sample
for i, row in tqdm(df.iterrows()): 
    # Read string list from pkl file
    pkl_path = os.path.join('rob_str', df.loc[i,'goldID']+'.pkl') 
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
for i in range(len(df)):
    df.loc[i, 'project'] = re.sub(r'\d+', "", df.loc[i, 'goldID'])


# Mismatch
sub = df
#sub = df[df.project == 'iicarus']
#df_mis = sub[sub.rgx_random != sub.RandomizationTreatmentControl]; print(len(df_mis)) 
#df_mis = sub[sub.rgx_blind != sub.BlindedOutcomeAssessment]; print(len(df_mis)) 
#df_mis = sub[sub.rgx_sample != sub.SampleSizeCalculation]; print(len(df_mis)) 
#df_mis = sub[(sub['ConflictsOfInterest'].isnull() == False) & (sub.rgx_conflict != sub.ConflictsOfInterest)]; print(len(df_mis)) 
#df_mis = sub[(sub['AnimalWelfareRegulations'].isnull() == False) & (sub.rgx_welfare != sub.AnimalWelfareRegulations)]; print(len(df_mis)) 
#
#df_mis = sub[(sub.rgx_random != sub.RandomizationTreatmentControl) | 
#             (sub.rgx_blind != sub.BlindedOutcomeAssessment) |
#             (sub.rgx_sample != sub.SampleSizeCalculation) |
#             ((sub['ConflictsOfInterest'].isnull() == False) & (sub.rgx_conflict != sub.ConflictsOfInterest)) |
#             ((sub['AnimalWelfareRegulations'].isnull() == False) & (sub.rgx_welfare != sub.AnimalWelfareRegulations))]

# rand/blind/ssc mismatched at the same time
df_mis = sub[(sub.rgx_random != sub.RandomizationTreatmentControl) & 
             (sub.rgx_blind != sub.BlindedOutcomeAssessment) &
             (sub.rgx_sample != sub.SampleSizeCalculation)]

# Every two of rand/blind/ssc mismatched
df_mis = sub[((sub.rgx_random != sub.RandomizationTreatmentControl) & (sub.rgx_blind != sub.BlindedOutcomeAssessment)) |
             ((sub.rgx_random != sub.RandomizationTreatmentControl) & (sub.rgx_sample != sub.SampleSizeCalculation)) |
             ((sub.rgx_blind != sub.BlindedOutcomeAssessment) & (sub.rgx_sample != sub.SampleSizeCalculation))]

ids = list(df_mis['goldID'])
id_common = list(set(id_checked) & set(ids))  # len = 24
# Removed records checked by MM
df_fin = df_mis[-df_mis["goldID"].isin(id_common)]  # len = 496-24 = 472

# Info list of all gold data
gold_info = read_json(json_path='rob_info.json')
err_list = [g for g in gold_info if g['goldID'] in list(df_fin['goldID'])]


# Create data frame for SyRF          
err_df = pd.DataFrame(columns = ['Title', 'Authors', 'Publication Name', 'Alternate Name', 'Abstract',
                                 'Url', 'Author Address', 'Year', 'DOI', 
                                 'Keywords', 'Reference Type', 'PDF Relative Path'])

for g in err_list:
    err_df = err_df.append({'Title': "", 'Authors': "", 'Publication Name': "", 'Alternate Name': "", 'Abstract': "",
                            'Url': "", 'Author Address': "", 'Year': "", 'DOI': "", 
                            'Keywords': g['goldID'], 'Reference Type': g['goldID'], 'PDF Relative Path': g['DocumentLink'],
                            'RandomizationTreatmentControl': g['RandomizationTreatmentControl'],
                            'AllocationConcealment': g['AllocationConcealment'],
                            'BlindedOutcomeAssessment': g['BlindedOutcomeAssessment'],
                            'SampleSizeCalculation': g['SampleSizeCalculation'],
                            'AnimalWelfareRegulations': g['AnimalWelfareRegulations'],
                            'ConflictsOfInterest': g['ConflictsOfInterest'],
                            'AnimalExclusions': g['AnimalExclusions']
                            }, ignore_index=True)
    
    
# Modify Reference type
err_df['Reference Type'] = err_df['Reference Type'].str.replace(r'[0-9]', '')  

# Copy PDFs to 'error_review' folder
err_df['PDF Relative Path'] = err_df['PDF Relative Path'].str.replace('psycho/', "data/psycho/")

os.chdir('/media/mynewdrive/rob')
for i, row in err_df.iterrows():  
    old_path = err_df.loc[i,'PDF Relative Path']            
    new_path = os.path.join('data/error_review', err_df.loc[i,'Reference Type'] , os.path.basename(old_path))
    err_df.loc[i,'PDF Relative Path'] = new_path    
    if os.path.exists(old_path) == False:
        print('PDF does not exist: {}'.format(old_path))
    else:        
        shutil.copy2(old_path, new_path)


err_df.to_csv('data/error_review/rob_error_check_regex.csv', sep=',', encoding='utf-8')


    


