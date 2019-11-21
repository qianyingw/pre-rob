#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 18:05:06 2019

@author: qwang
"""

import os
import json
import random
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
random.shuffle(gold)
with open('rob_gold_tokens.json', 'w') as fout:
    for dic in gold:     
        fout.write(json.dumps(dic) + '\n')   
    

