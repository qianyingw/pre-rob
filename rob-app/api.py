#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 22:03:40 2020

@author: qwang
"""

from flask import Flask, request
from flask_restful import Resource, Api
import os

import sys
import re
import json
from pred import pred_prob



             
             
             
#%%
PROB_PATH = {
    'arg-r': 'pth/awr_13.json',
    'pth-r': 'pth/awr_13.pth.tar',
    'fld-r': 'fld/awr_13.Field',
    
    'arg-b': 'pth/awb_32.json',
    'pth-b': 'pth/awb_32.pth.tar',
    'fld-b': 'fld/awb_32.Field',
    
    'arg-i': 'pth/cwi_6.json',
    'pth-i': 'pth/cwi_6.pth.tar',
    'fld-i': 'fld/cwi_6.Field',
    
    'arg-w': 'pth/cww_15.json',
    'pth-w': 'pth/cww_15.pth.tar',
    'fld-w': 'fld/cww_15.Field',
    
    'arg-e': 'pth/awe_8.json',
    'pth-e': 'pth/awe_8.pth.tar',
    'fld-e': 'fld/awe_8.Field',
}

class PreRob():
    def __init__(self, prob_path, txt_dir):      
    
        for key, value in prob_path.items():
            prob_path[key] = os.path.join(os.getcwd(), value)
        self.prob_path = prob_path
        self.txt_dir = txt_dir
        
    def get_txt_path(self):
        txt_paths = []
        for root, _, files in os.walk(self.txt_dir):            
            for f in files:
                if f.endswith(".txt"):
                    txt_paths.append(os.path.join(root, f))
        self.txt_paths = txt_paths
    
    def process_text(self, text):       
        text = re.sub(r"^(?:[\t ]*(?:\r?\n|\r))+", " ", text, flags=re.MULTILINE)  # Remove emtpy lines        
        text = text.encode("ascii", errors="ignore").decode()  # Remove non-ascii characters         
        text = re.sub(r'\s+', " ", text)  # Strip whitespaces     
        text = re.sub(r'^[\s]', "", text)  # Remove the whitespace at start and end of line
        text = re.sub(r'[\s]$', "", text)
        return text
          
    def pred_probs(self): 
        output = []
        Id = 0
        for path in self.txt_paths:
            with open(path, 'r', encoding='utf-8', errors='ignore') as fin:
                text = fin.read()           
            text = self.process_text(text)    
            
            pr = pred_prob(self.prob_path['arg-r'], self.prob_path['fld-r'], self.prob_path['pth-r'], text).astype(float)
            pb = pred_prob(self.prob_path['arg-b'], self.prob_path['fld-b'], self.prob_path['pth-b'], text).astype(float)
            pi = pred_prob(self.prob_path['arg-i'], self.prob_path['fld-i'], self.prob_path['pth-i'], text).astype(float)
            pw = pred_prob(self.prob_path['arg-w'], self.prob_path['fld-w'], self.prob_path['pth-w'], text).astype(float)
            pe = pred_prob(self.prob_path['arg-e'], self.prob_path['fld-e'], self.prob_path['pth-e'], text).astype(float)
            
            score = {"txt_id": Id, "txt_path": path,
        			 "random": pr,
        			 "blind": pb,
        			 "interest": pi,
        			 "welfare": pw,
        			 "exclusion": pe}
            Id = Id + 1
            output.append(score)
        return output
             

#%%
app = Flask(__name__)
api = Api(app)

dirs = {}

class Dir(Resource):
    def get(self, dir_id):
        return {dir_id: dirs[dir_id]}

    def put(self, dir_id):
        txt_dir = request.form['data']
        if os.path.isabs(txt_dir) == False:
            txt_dir = os.path.join(os.getcwd(), txt_dir)         
        if os.path.isdir(txt_dir) == False:
            txt_dir = 'Invalid path'
        
        rober = PreRob(PROB_PATH, txt_dir)
        rober.get_txt_path()
        output = rober.pred_probs()
    
        dirs[dir_id] = txt_dir
        out = {dir_id: dirs[dir_id], 'dir_output': output}
        
        with open(os.path.join(txt_dir, 'scores.json'), 'w') as fp:
            json.dump(out, fp)

        return out
    
api.add_resource(Dir, '/<string:dir_id>')

if __name__ == '__main__':
    # app.run(debug=False)
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
