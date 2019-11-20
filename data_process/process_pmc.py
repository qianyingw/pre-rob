#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 16:30:17 2019

@author: qwang
"""

import os
import re
import pandas as pd
import json
from tqdm import tqdm

# change working directory
wdir = '/home/qwang/rob/'
os.chdir(wdir)

import src.data_process.regex as regex
import src.data_process.tokenizer as tokenizer


#%% Data and setting
pmc = pd.read_csv("data/pmc/AnimalStudiesFile.csv", sep=',', engine="python", encoding="utf-8")   
list(pmc.columns)
pmc = pmc[['Id', 'ITEM_ID', 'filePath', 'filePathFull', 'score', 'textFilePath']]
pmc.shape  # (96546, 6)


# Modify file path
pmc['nxmlPath'] = pmc['filePathFull'].str.replace('output', 'data/pmc/nxml')
pmc['txtPath'] = pmc['textFilePath'].str.replace('output', 'data/pmc/nxml')
pmc = pmc[['Id', 'ITEM_ID', 'nxmlPath', 'txtPath', 'score']]

# Modify three file paths which are incorrect from the index file 'AnimalStudiesFile.csv'
pmc.loc[pmc.nxmlPath=='data/pmc/nxml/oa_bulk/Eneuro/PMC4443438.nxml', 'nxmlPath'] = 'data/pmc/nxml/oa_bulk/eNeuro/PMC4443438.nxml'
pmc.loc[pmc.txtPath=='data/pmc/nxml/oa_bulk/Eneuro/PMC4443438.txt', 'txtPath'] = 'data/pmc/nxml/oa_bulk/eNeuro/PMC4443438.txt'

pmc.loc[pmc.nxmlPath=='data/pmc/nxml/oa_bulk/Peerj/PMC3628383.nxml', 'nxmlPath'] = 'data/pmc/nxml/oa_bulk/PeerJ/PMC3628383.nxml'
pmc.loc[pmc.txtPath=='data/pmc/nxml/oa_bulk/Peerj/PMC3628383.txt', 'txtPath'] = 'data/pmc/nxml/oa_bulk/PeerJ/PMC3628383.txt'

pmc.loc[pmc.nxmlPath=='data/pmc/nxml/oa_bulk/Peerj/PMC3628749.nxml', 'nxmlPath'] = 'data/pmc/nxml/oa_bulk/PeerJ/PMC3628749.nxml'
pmc.loc[pmc.txtPath=='data/pmc/nxml/oa_bulk/Peerj/PMC3628749.txt', 'txtPath'] = 'data/pmc/nxml/oa_bulk/PeerJ/PMC3628749.txt'



regex_dir = 'regex'
json_dir = 'data/pmc/json'



#%%

def dict2json(d, regex_dir, json_dir):
    """
        d: single dictionary of one record from pmc dataframe
        regex_dir: directory containing regex txt files 
                RandomizationTreatmentControl.txt
                BlindedOutcomeAssessment.txt
                SampleSizeCalculation.txt
                ConflictsOfInterest.txt
                AnimalWelfareRegulations.txt
    """
    if os.path.isfile(d["txtPath"]):
        with open(d['txtPath'], 'r', encoding='utf-8') as fin:
            text = fin.read()
        text_processed = tokenizer.preprocess_text(text)
        
        # Add regex decision labels
        regex_str = regex.read_regex(os.path.join(regex_dir, 'RandomizationTreatmentControl.txt'))
        d['RandomizationTreatmentControl'] = regex.doc_annotate(regex_str, text_processed)
        
        regex_str = regex.read_regex(os.path.join(regex_dir, 'BlindedOutcomeAssessment.txt'))
        d['BlindedOutcomeAssessment'] = regex.doc_annotate(regex_str, text_processed)
        
        regex_str = regex.read_regex(os.path.join(regex_dir, 'SampleSizeCalculation.txt'))
        d['SampleSizeCalculation'] = regex.doc_annotate(regex_str, text_processed)
        
        regex_str = regex.read_regex(os.path.join(regex_dir, 'ConflictsOfInterest.txt'))
        d['ConflictsOfInterest'] = regex.doc_annotate(regex_str, text_processed)
        
        regex_str = regex.read_regex(os.path.join(regex_dir, 'AnimalWelfareRegulations.txt'))
        d['AnimalWelfareRegulations'] = regex.doc_annotate(regex_str, text_processed)
        
        # Add tokens
        d['textTokens'] = tokenizer.tokenize_text(text_processed)
        
        # Covert dictionary list to json
        with open(d['jsonPath'], 'w', encoding='utf-8') as fout:
            fout.write(json.dumps(d))
    else:
        print(d['txtPath'])
        

#%%
pm = pmc[:10000]
pm = pmc[10000:20000]
pm = pmc[20000:30000]
pm = pmc[30000:40000]  # 9999
pm = pmc[40000:50000]  # 9999
pm = pmc[50000:60000]  # 9999
pm = pmc[60000:70000]
pm = pmc[70000:80000]
pm = pmc[80000:90000]

pm = pmc[90000:]
with tqdm(total=len(pm)) as progress_bar:
    for _, df in pm.iterrows():     
        jsonPath = os.path.join(json_dir, re.sub('.nxml', '.json', os.path.basename(df.nxmlPath)))
        dic = {'Id': df.Id,
              'ITEM_ID': df.ITEM_ID,
              'nxmlPath': df.nxmlPath,
              'txtPath': df.txtPath,
              'jsonPath': jsonPath,
              'score': df.score}
    
        
        dict2json(dic, regex_dir, json_dir)    
        progress_bar.update(1)
    



#jsonPath = 'data/pmc/json/PMC4443438.json'
#dic = {'Id': 677703,
#      'ITEM_ID': 'PMC4443438',
#      'nxmlPath': 'data/pmc/nxml/oa_bulk/eNeuro/PMC4443438.nxml',
#      'txtPath': 'data/pmc/nxml/oa_bulk/eNeuro/PMC4443438.txt',
#      'jsonPath': jsonPath,
#      'score': 0.857261}
#        
#temp = pmc.loc[pmc.nxmlPath=='data/pmc/nxml/oa_bulk/PeerJ/PMC3628383.nxml']
#jsonPath = 'data/pmc/json/PMC3628383.json'
#dic = {'Id': 1818128,
#      'ITEM_ID': 'PMC3628383',
#      'nxmlPath': 'data/pmc/nxml/oa_bulk/PeerJ/PMC3628383.nxml',
#      'txtPath': 'data/pmc/nxml/oa_bulk/PeerJ/PMC3628383.txt',
#      'jsonPath': jsonPath,
#      'score': 0.873513}
#        
#temp = pmc.loc[pmc.nxmlPath=='data/pmc/nxml/oa_bulk/PeerJ/PMC3628749.nxml']
#jsonPath = 'data/pmc/json/PMC3628749.json'
#dic = {'Id': 1818145,
#      'ITEM_ID': 'PMC3628749',
#      'nxmlPath': 'data/pmc/nxml/oa_bulk/PeerJ/PMC3628749.nxml',
#      'txtPath': 'data/pmc/nxml/oa_bulk/PeerJ/PMC3628749.txt',
#      'jsonPath': jsonPath,
#      'score': 0.870928}
        
