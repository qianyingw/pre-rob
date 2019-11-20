#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
From CS230 Code Examples

Created on Fri Oct  4 11:56:35 2019
@author: qwang
"""

import json
import logging
import os
import shutil

import torch



class Params():
    """
    Loads hyperparameters from a json file
    Example:
        ```
        par = Parmas(json_path)
        print(par.filter_sizes)
        par.dropout = 0.1  # change the value of dropout rate in pars
        par.__dict__
            {'seed': 1234,
             'batch_size': 32,
             ...
             'embedding_dim': 200}
        ```       
    """
    
    def __init__(self, json_path):
        
        with open(json_path) as fin:
            params = json.load(fin)
            
        self.seed = params['seed']
        self.batch_size = params['batch_size']
        self.n_epochs = params['n_epochs']   
        self.train_ratio = params['train_ratio']
        self.val_ratio = params['val_ratio']
        self.max_vocab_size = params['max_vocab_size']
        self.min_occur_freq = params['min_occur_freq']
        self.n_filters = params['n_filters']
        self.filter_sizes = params['filter_sizes']
        self.dropout = params['dropout']
        self.embedding_dim = params['embedding_dim']
        
        
    def update(self, json_path):        
        with open(json_path) as fin:
            params = json.load(fin)            
        self.__dict__.update(params)
                    
    
    def save(self, json_path):
        with open(json_path, 'w') as fout:
            json.dump(self.__dict__, fout, indent=4)
            
            
            

# class RunningAverage()



def set_logger(log_path):
    """
    Set the logger to log info in terminal and file 'log_path'
        Loggers expose the interface that application code directly uses.
        Handlers send the log records (created by loggers) to the appropriate destination.
        Formatters specify the layout of log records in the final output.
    Exmaple:
        ```
        logging.info("Start training...")
        ```
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        # Log to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)
        
        # Log to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


        
def save_dict_to_json(d, json_path):
    """
    Save dict of floats to json file
    d: dict of float-castable values (np.float, int, float, etc.)
      
    """      
    with open(json_path, 'w') as fout:
        d = {key: float(value) for key, value in d.items()}
        json.dump(d, fout, indent=4)
        
        
        
def save_checkpoint(state, is_best, checkdir):
    """
    Save model and training parameters at checkpoint + 'last.pth.tar'. 
    If is_best==True, also saves checkpoint + 'best.pth.tar'
    Params:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkdir: (string) folder where parameters are to be saved
    """        
    filepath = os.path.join(checkdir, 'last.pth.tar')
    if os.path.exists(checkdir) == False:
        os.mkdir(checkdir)
    torch.save(state, filepath)    
    
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkdir, 'best.pth.tar'))
        
        
        
def load_checkpoint(checkfile, model, optimizer=None):
    """
    Load model parameters (state_dict) from checkfile. 
    If optimizer is provided, loads state_dict of optimizer assuming it is present in checkpoint.
    Params:
        checkfile: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """        
    if os.path.exists(checkfile) == False:
        raise("File doesn't exist {}".format(checkfile))
    checkfile = torch.load(checkfile)
    model.load_state_dict(checkfile['state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkfile['optim_dict'])
    
    return checkfile
    
        
        
        
        
        
        
        
    