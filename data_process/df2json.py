# -*- coding: utf-8 -*-
"""
Convert RoB Info df to dict lists
Add 'textTokens' element to each dict list
Output to json

Input: 
    df_info: RoB info dataframe which contains at least following columns
                  'goldID', 'fileLink','DocumentLink','txtLink'.
             For gold data, it is supposed to be
                     df_info[['goldID',
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
   
    json_path: absolute path of final json file

* df_info['txtLink'] are relative paths under 'U:/Datastore/CMVM/scs/groups/DCN/TRIALDEV/CAMARADES/Qianying/rob/'

Created on Fri Sep 27 11:10:39 2019
@author: qwang
"""

import json

from tqdm import tqdm
from src.data_process.tokenizer import preprocess_text, text_tokenizer, sent_encoder


#%%
def df2json(df_info, json_path):
   
    dict_list = df_info.to_dict('records')

    # Add fullText and textTokens to each text
    for i, dic in tqdm(enumerate(dict_list)):
        txt_path = dic['txtLink']
        try:
            with open(txt_path, 'r', encoding='utf-8') as fp:
                text = fp.read()
            text_processed = preprocess_text(text)        
            # dict_list[i]['fullText'] = text_processed
            dict_list[i]['sentTokens'], dict_list[i]['wordTokens'] = text_tokenizer(text_processed)
        except:
            dict_list[i]['sentTokens'] = ''
            dict_list[i]['wordTokens'] = ''
                 
    # Covert dictionary list to json
    with open(json_path, 'w') as fout:
        for dic in dict_list:     
            fout.write(json.dumps(dic) + '\n')
        
#%% For sentence encoder
def df2json_embed(df_info, json_path, embed_func):
   
    dict_list = df_info.to_dict('records')
    # Add document matrix
    for i, dic in tqdm(enumerate(dict_list)):
        txt_path = dic['txtLink']
        try:
            with open(txt_path, 'r', encoding='utf-8') as fp:
                text = fp.read()
            text_processed = preprocess_text(text)        
            dict_list[i]['docMat'] = sent_encoder(embed_func, text_processed)
        except:
            dict_list[i]['docMat'] = ''
                 
    # Covert dictionary list to json
    with open(json_path, 'w') as fout:
        for dic in dict_list:     
            fout.write(json.dumps(dic) + '\n')    
        