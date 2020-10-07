#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 17:43:36 2020

@author: qwang
"""


import os
import json
from tqdm import tqdm
import torch
import torch.nn as nn

import utils

#%% Train

def train_fn(model, data_loader, optimizer, scheduler, loss_fn, metrics_fn, device, clip, accum_step, threshold):
    
    scores = {'loss': 0, 'accuracy': 0, 'f1': 0, 'recall': 0, 'precision': 0, 'specificity': 0}
    len_iter = len(data_loader)
    
    model.train()
    
    optimizer.zero_grad()
    with tqdm(total=len_iter) as progress_bar:
        for i, batch in enumerate(data_loader):
            
            batch_doc, batch_label = batch
            batch_doc = batch_doc.to(device)
            batch_label = batch_label.to(device)                   
            
            preds = model(batch_doc)  # preds.shape = [batch_size, num_labels]
            
            loss = loss_fn(preds, batch_label)    
            scores['loss'] += loss.item()
            epoch_scores = metrics_fn(preds, batch_label, threshold)  # dictionary of 5 metric scores
            for key, value in epoch_scores.items():               
                scores[key] += value  
            
            loss = loss / accum_step  # loss gradients are accumulated by loss.backward() so we need to ave accumulated loss gradients
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)  # prevent exploding gradients
                      
            # Gradient accumulation    
            if (i+1) % accum_step == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            # Update progress bar                          
            progress_bar.update(1)  
    
    for key, value in scores.items():
        scores[key] = value / len_iter   
    return scores



#%% Evaluate   
def valid_fn(model, data_loader, loss_fn, metrics_fn, device, threshold):
    
    scores = {'loss': 0, 'accuracy': 0, 'f1': 0, 'recall': 0, 'precision': 0, 'specificity': 0}
    len_iter = len(data_loader)
    model.eval()

    with torch.no_grad():
        with tqdm(total=len_iter) as progress_bar:
            for batch in data_loader:
                
                batch_doc, batch_label = batch
                batch_doc = batch_doc.to(device)
                batch_label = batch_label.to(device)                   
                    
                preds = model(batch_doc)
                
                loss = loss_fn(preds, batch_label) 
                epoch_scores = metrics_fn(preds, batch_label, threshold)
                
                scores['loss'] += loss.item()
                for key, value in epoch_scores.items():               
                    scores[key] += value        
                progress_bar.update(1)  # update progress bar   
                
    for key, value in scores.items():
        scores[key] = value / len_iter   

    return scores


        

#%%
# def evaluate(model, data_loader, criterion, metrics, device, threshold):
def test_fn(model, test_loader, loss_fn, metrics_fn, args, device, restore_file):   
     
    utils.load_checkpoint(os.path.join(args.exp_dir, restore_file + '.pth.tar'), model) 
    test_scores = valid_fn(model, test_loader, loss_fn, metrics_fn, device, args.threshold)
    save_path = os.path.join(args.exp_dir, "test_scores.json")
    utils.save_dict_to_json(test_scores, save_path)  
    print('\n[Test] loss: {0:.3f} | acc: {1:.2f}% | f1: {2:.2f}% | recall: {3:.2f}% | precision: {4:.2f}% | specificity: {5:.2f}%'.format(
            test_scores['loss'], test_scores['accuracy']*100, test_scores['f1']*100, test_scores['recall']*100, test_scores['precision']*100, test_scores['specificity']*100))
    
    return test_scores