#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 14:33:15 2021

@author: qwang
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 18:16:02 2020

@author: qwang
"""

import os
import re

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd

# import rob_fn
from rob_fn import load_model, load_model_bert, pred, pred_bert, extract_sent
          
#%%
class PreRob():
    def __init__(self, txt_info):      
    
        # for key, value in prob_path.items():
        #     prob_path[key] = os.path.join(os.getcwd(), value)
        # self.prob_path = prob_path
        # self.sent_path = sent_path
        self.txt_info = txt_info
        self.txt_paths = []
        self.ids = []
        
    def get_txt_path(self):       
        txt_info = self.txt_info
        txt_paths = self.txt_paths
        
        # If 'txt_info' is a folder
        if os.path.isdir(txt_info) == True:
            # txt_info = os.path.join(os.getcwd(), txt_info)      
            for root, _, files in os.walk(txt_info):            
                for f in files:
                    if f.endswith(".txt"):
                        txt_paths.append(os.path.join(root, f))
               
        # if 'txt_info' is a single txt path
        if os.path.exists(txt_info) == True and txt_info.endswith(".txt") == True:
            txt_paths.append(txt_info)
        
        # if 'txt_info' is a string containing multiple txt paths       
        if len(txt_info.split(".txt,")) > 1:
            paths = txt_info.split(".txt,")
            for p in paths[:-1]:
                txt_paths.append(p+'.txt')
            txt_paths.append(paths[-1])   
            
        # If 'txt_info' is the path of a csv containing relative paths
        if os.path.exists(txt_info) == True and txt_info.endswith(".csv") == True:
            path_df = pd.read_csv(txt_info, sep=',')    
            for i, row in path_df.iterrows():   
                txt_path = os.path.join(os.path.dirname(txt_info), path_df.loc[i,'path'])
                if os.path.exists(txt_path):
                    txt_paths.append(txt_path)          
                    self.ids.append(path_df.loc[i,'id'])
            
        self.txt_paths = txt_paths
    
    
    def process_text(self, text):       
        # text = re.sub(r"^(?:[\t ]*(?:\r?\n|\r))+", " ", text, flags=re.MULTILINE)  # Remove emtpy lines        
        # text = text.encode("ascii", errors="ignore").decode()  # Remove non-ascii characters         
        # text = re.sub(r'\s+', " ", text)  # Strip whitespaces     
        # text = re.sub(r'^[\s]', "", text)  # Remove the whitespace at start and end of line
        # text = re.sub(r'[\s]$', "", text)
        
        # Remove texts before the first occurence of 'Introduction' or 'INTRODUCTION'
        text = re.sub(r".*?(Introduction|INTRODUCTION)\s{0,}\n{1,}", " ", text, count=1, flags=re.DOTALL)    
        # Remove reference after the last occurence 
        s = re.search(p_ref, text)
        if s: text = s[0]  
        # Remove citations 
        text = re.sub(r"\s+[\[][^a-zA-Z]+[\]]", "", text)
        # Remove links
        text = re.sub(r"https?:/\/\S+", " ", text)
        # Remove emtpy lines
        text = re.sub(r"^(?:[\t ]*(?:\r?\n|\r))+", " ", text, flags=re.MULTILINE)
        # Remove lines with digits/(digits,punctuations,line character) only
        text = re.sub(r"^\W{0,}\d{1,}\W{0,}$", "", text)
        # Remove non-ascii characters
        text = text.encode("ascii", errors="ignore").decode()     
        # Strip whitespaces 
        text = re.sub(r'\s+', " ", text)
        # Remove the whitespace at start and end of line
        text = re.sub(r'^[\s]', "", text)
        text = re.sub(r'[\s]$', "", text)
    
        return text 
          
    def pred_probs(self, num_sents=0): 
        if self.txt_paths == []:
            output = {"message": "Folder/TXTs not found"}  # folder doesn't exist or no txt files found in the folder
            
        else:    
            output = []   
            co = 0
            for path in self.txt_paths:

                if os.path.isabs(path) == False:
                    path = os.path.join(os.getcwd(), path)
                with open(path, 'r', encoding='utf-8', errors='ignore') as fin:
                    text = fin.read()           
                text = self.process_text(text)   
                
                pr = pred(text, mod1, arg1, TEXT1).astype(float)
                pb = pred(text, mod2, arg2, TEXT2).astype(float)
                pi = pred(text, mod3, arg3, TEXT3).astype(float)
                pw = pred_bert(text, mod4, rob_sent, max_n_sent=30).astype(float)
                pe = pred(text, mod5, arg5, TEXT5).astype(float)
                
                co += 1
                print('{} files done.'.format(co))
                score = {"txt_path": path,
            			 "random": pr, "blind": pb, "interest": pi, "welfare": pw, "exclusion": pe}
                
                if num_sents > 0: 
                    sr = extract_sent(text, smod1, sarg1, sTEXT1, num_sents)
                    sb = extract_sent(text, smod2, sarg2, sTEXT2, num_sents)
                    si = extract_sent(text, smod3, sarg3, sTEXT3, num_sents)
                    sw = extract_sent(text, smod4, sarg4, sTEXT4, num_sents)
                    se = extract_sent(text, smod5, sarg5, sTEXT5, num_sents)
                    score['sentences'] = {"random": sr, "blind": sb, "interest": si, "welfare": sw, "exclusion": se}             
                              
                output.append(score)
                
            for i, _ in enumerate(output):
                if self.ids is None:
                    output[i]['id'] = str(i + 1)
                else:
                    output[i]['id'] = str(self.ids[i])
                               
        return output


#%%
import argparse
parser = argparse.ArgumentParser(description='Get CSV input')
parser.add_argument('-p', "--csv", nargs="?", type=str, default=None, help='Absolute path of csv input file')
parser.add_argument('-s', "--sent", nargs="?", type=int, default=0, help='Number of sentences extracted')

args = parser.parse_args()
txt_info = args.csv
# txt_info = os.path.join(os.getcwd(), args.csv)
num_sents = int(args.sent)

p_ref = re.compile(r"(.*Reference\s{0,}\n)|(.*References\s{0,}\n)|(.*Reference list\s{0,}\n)|(.*REFERENCE\s{0,}\n)|(.*REFERENCES\s{0,}\n)|(.*REFERENCE LIST\s{0,}\n)", 
                   flags=re.DOTALL)

mod1, arg1, TEXT1 = load_model(arg_path='pth/awr_13.json', pth_path='pth/awr_13.pth.tar', fld_path='pth/awr_13.Field')
mod2, arg2, TEXT2 = load_model(arg_path='pth/awb_32.json', pth_path='pth/awb_32.pth.tar', fld_path='pth/awb_32.Field')
mod3, arg3, TEXT3 = load_model(arg_path='pth/cwi_6.json', pth_path='pth/cwi_6.pth.tar', fld_path='pth/cwi_6.Field')
mod4, rob_sent = load_model_bert(arg_path='pth/dsc_w0.json', pth_path='pth/dsc_w0.pth.tar')
mod5, arg5, TEXT5 = load_model(arg_path='pth/awe_8.json', pth_path='pth/awe_8.pth.tar', fld_path='pth/awe_8.Field')

if num_sents:
    smod1, sarg1, sTEXT1 = load_model(arg_path='pth/hr_4.json', pth_path='pth/hr_4.pth.tar', fld_path='pth/hr_4.Field')
    smod2, sarg2, sTEXT2 = load_model(arg_path='pth/hb_5.json', pth_path='pth/hb_5.pth.tar', fld_path='pth/hb_5.Field')
    smod3, sarg3, sTEXT3 = load_model(arg_path='pth/hi_4.json', pth_path='pth/hi_4.pth.tar', fld_path='pth/hi_4.Field')
    smod4, sarg4, sTEXT4 = load_model(arg_path='pth/hw_17.json', pth_path='pth/hw_17.pth.tar', fld_path='pth/hw_17.Field')
    smod5, sarg5, sTEXT5 = load_model(arg_path='pth/he_26.json', pth_path='pth/he_26.pth.tar', fld_path='pth/he_26.Field')
    
      
if txt_info and txt_info.endswith(".csv") == True:
    rober = PreRob(txt_info)
    rober.get_txt_path() 
    if num_sents:
        output = rober.pred_probs(num_sents)
    else:    
        output = rober.pred_probs()
    output_df = pd.DataFrame(output)
    csv_dir = os.path.dirname(txt_info)
    output_df.to_csv(os.path.join(csv_dir, 'output.csv'), sep=',', encoding='utf-8')
else:
    print("The input file is not csv")
    
                  

