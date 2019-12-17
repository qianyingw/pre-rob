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
random.seed(1234)

#%% Read json files
os.chdir('/media/mynewdrive/rob/data')

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


#%% Merge data
gold = stroke + psy + neuro + npqip + iicarus  # len(gold)=7904


#%% Remove records with too short token length
goldID_del = []
num_tokens = []

for g in gold:
    if len(g['textTokens']) < 2000:
        goldID_del.append(g['goldID'])
    else:
        num_tokens.append(len(g['textTokens']))
        
gold = [g for g in gold if g['goldID'] not in goldID_del]  # 7872

#max(num_tokens)  # 27799
#min(num_tokens)  # 2005

#%% Output
random.shuffle(gold)
with open('rob_gold_tokens.json', 'w') as fout:
    for dic in gold:     
        fout.write(json.dumps(dic) + '\n')   
    
    
   
#%% Generate error checking records
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
           
           
# Create data frame for error checking           
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

    





