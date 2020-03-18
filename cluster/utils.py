#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 16:55:35 2019
From CS230 Code Examples
@author: qwang
"""

import os
import logging
import shutil
import torch
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% Logging
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
def metrics(preds, y, th=0.5):
    """
    Params:
        preds: torch tensor, [batch_size, output_dim]
        y: torch tensor, [batch_size]
        th: threshold (default=0.5)      
    Yields:
        A dictionary of accuracy, f1 score, recall, precision and specificity       
        
    """   
    # y_preds = preds.argmax(dim=1, keepdim=False)  # [batch_size, output_dim]  --> [batch_size]
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



#%% Plot performance 
def plot_prfs(prfs_json_path):
    
    with open(prfs_json_path) as f:
        dat = json.load(f)
        
    # Create scores dataframe
    epochs = int(len(dat['prfs'])/2)
    train_df = pd.DataFrame(columns=['Loss', 'Accuracy', 'F1', 'Recall', 'Precision', 'Specificity'])
    valid_df = pd.DataFrame(columns=['Loss', 'Accuracy', 'F1', 'Recall', 'Precision', 'Specificity'])
    for i in range(epochs):
        train_df.loc[i] = list(dat['prfs']['train_'+str(i+1)].values())
        valid_df.loc[i] = list(dat['prfs']['valid_'+str(i+1)].values()) 
    
    # Plot
    plt.figure(figsize=(15,5))
    x = np.arange(len(train_df)) + 1   
    # Loss / F1
    plt.subplot(1, 2, 1)
    plt.title("Loss and F1")
    plt.plot(x, train_df['Loss'], label="train_loss", color='C5')
    plt.plot(x, valid_df['Loss'], label="val_loss", color='C5', linestyle='--')
    plt.plot(x, train_df['F1'], label="train_f1", color='C9')
    plt.plot(x, valid_df['F1'], label="val_f1", color='C9', linestyle='--')
    plt.xticks(np.arange(2, len(x)+2, step=2))
    plt.legend(loc='upper right')
    # Accuracy / Recall
    plt.subplot(1, 2, 2)
    plt.title("Accuracy and Score")
    plt.plot(x, train_df['Accuracy'], label="train_acc", color='C0', alpha=0.8)
    plt.plot(x, valid_df['Accuracy'], label="val_acc", color='C0', linestyle='--', alpha=0.8)
    #plt.plot(x, train_df['F1'], label="train_f1", color='C9')
    #plt.plot(x, valid_df['F1'], label="val_f1", color='C9', linestyle='--')
    plt.plot(x, train_df['Recall'], label="train_rec", color='C1', alpha=0.8)
    plt.plot(x, valid_df['Recall'], label="val_rec", color='C1', linestyle='--', alpha=0.8)
    plt.xticks(np.arange(2, len(x)+2, step=2))
    plt.legend(loc='lower right')    
    
    # Save png
    output_dir = os.path.dirname(prfs_json_path)
    plt.savefig(os.path.join(output_dir, 'prfs.png'))