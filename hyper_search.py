#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 19:25:44 2019

@author: qwang
"""

import argparse
import os
from subprocess import check_call
import sys
import utils
import shutil

PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--wdir', default='/home/qwang/rob', help='Project working directory')
parser.add_argument('--parent_dir', default='src/experiment/dropout', help='Directory containing params.json')
parser.add_argument('--data_dir', default='data/psycho', help='Directory containing the dataset')              




def launch_job(model_dir, data_dir, params):
    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)
    
    # Launch training with this config
    cmd = "{python} src/train.py --model_dir={model_dir} --data_dir={data_dir}".format(python=PYTHON, model_dir=model_dir, data_dir=data_dir)
    print(cmd)
    
    check_call(cmd, shell=True)

if __name__ == "__main__":
    # Load "reference" parameters from parent_dir json file
    args = parser.parse_args()
    os.chdir(args.wdir)
    
    json_path = os.path.join(args.parent_dir, "params.json")
    assert os.path.isfile(json_path), "No configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    
    # Perform hypersearch over one parameter
    dropout_rates = [0.1, 0.5, 0.9]
    
    for dropout_rate in dropout_rates:
        # Modify the corresponding 
        params.dropout = dropout_rate        
              
        # Create a new folder in parent_dir with unique 'job_name'
        job_name = "dropout_{}".format(dropout_rate)
        model_dir = os.path.join(args.parent_dir, job_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Copy 'reference' params.json to each 'job_name' directory
        shutil.copyfile(json_path, os.path.join(args.parent_dir, job_name, 'params.json'))
        
        # Launch job
        launch_job(model_dir, args.data_dir, params)
        