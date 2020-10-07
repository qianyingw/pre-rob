#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 19:08:11 2020

@author: qwang
"""


import os
import shutil
import torch
import json      
        
#%%        
def save_dict_to_json(d, json_path):
    """
    Save dict of floats to json file
    d: dict of float-castable values (np.float, int, float, etc.)
      
    """      
    with open(json_path, 'w') as fout:
        d = {key: float(value) for key, value in d.items()}
        json.dump(d, fout, indent=4)
        
        
#%% Checkpoint        
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


#%% Metrics   
def metrics_fn(preds, y, th=0.5):
    """
    Params:
        preds: torch tensor, [batch_size, output_dim]
        y: torch tensor, [batch_size]
        
    Yields:
        A dictionary of accuracy, f1 score, recall, precision and specificity       
        
    """   
    
    #  y_preds = preds.argmax(dim=1, keepdim=False)  # [batch_size, output_dim]  --> [batch_size]
    if torch.cuda.device_count() == 1:
        # y_preds = (preds[:,1] > th).type(torch.ShortTensor).cuda()
        y_preds = (preds[:,1] > th).int().type(torch.LongTensor).cuda()
    else:
        # y_preds = (preds[:,1] > th).type(torch.ShortTensor)
        y_preds = (preds[:,1] > th).int().type(torch.LongTensor)
    
    ones = torch.ones_like(y_preds)
    zeros = torch.zeros_like(y_preds)
    
    pos = torch.eq(y_preds, y).sum().item()
    tp = (torch.eq(y_preds, ones) & torch.eq(y, ones)).sum().item()
    tn = (torch.eq(y_preds, zeros) & torch.eq(y, zeros)).sum().item()
    fp = (torch.eq(y_preds, ones) & torch.eq(y, zeros)).sum().item()
    fn = (torch.eq(y_preds, zeros) & torch.eq(y, ones)).sum().item()
    
    assert pos == tp + tn
    
    acc = pos / y.shape[0]  # torch.FloatTensor([y.shape[0]])
    f1 = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn != 0) else 0
    rec = tp / (tp + fn) if (tp + fn != 0) else 0
    ppv = tp / (tp + fp) if (tp + fp != 0) else 0
    spc = tn / (tn + fp) if (tn + fp != 0) else 0
    
    return {'accuracy': acc, 'f1': f1, 'recall': rec, 'precision': ppv, 'specificity': spc}