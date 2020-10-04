#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 22:06:23 2020

@author: qwang
"""

import torch
import torch.nn as nn
from tqdm import tqdm


softmax = nn.Softmax(dim=1)

#%%
def train_fn(model, data_loader, optimizer, scheduler, metrics_fn, clip, accum_step, threshold, device):
    
    scores = {'loss': 0, 'accuracy': 0, 'f1': 0, 'recall': 0, 'precision': 0, 'specificity': 0}
    len_iter = len(data_loader)
    
    model.train()
    optimizer.zero_grad()
    
    with tqdm(total=len_iter) as progress_bar:      
        for j, batch in enumerate(data_loader):                      
            
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            
            outputs = model(**inputs, labels=labels)
            
            loss = outputs.loss           
            scores['loss'] += loss.item() 
            
            logits = outputs.logits  # scores before softmax: [batch_size, output_dim]
            preds = softmax(logits)
            epoch_scores = metrics_fn(preds, labels, threshold)  # dictionary of 5 metric scores
            for key, value in epoch_scores.items():               
                scores[key] += value  
            
            loss = loss / accum_step  # average loss gradients bec they are accumulated by loss.backward()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            
            # Gradient accumulation    
            if (j+1) % accum_step == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            # Update progress bar                          
            progress_bar.update(1)  
    
    for key, value in scores.items():
        scores[key] = value / len_iter
        
    return scores
             

#%%
def valid_fn(model, data_loader, metrics_fn, device, threshold):
    
    scores = {'loss': 0, 'accuracy': 0, 'f1': 0, 'recall': 0, 'precision': 0, 'specificity': 0}
    len_iter = len(data_loader)
    model.eval()

    with torch.no_grad():
        with tqdm(total=len_iter) as progress_bar:
            for batch in data_loader:
                
                inputs = batch[0].to(device)
                labels = batch[1].to(device)
            
                outputs = model(**inputs, labels=labels)
                
                loss = outputs.loss           
                scores['loss'] += loss.item() 
                
                logits = outputs.logits  # scores before softmax: [batch_size, output_dim]
                preds = softmax(logits)
                epoch_scores = metrics_fn(preds, labels, threshold)  # dictionary of 5 metric scores
                for key, value in epoch_scores.items():               
                    scores[key] += value       
                progress_bar.update(1)  # update progress bar   
                
    for key, value in scores.items():
        scores[key] = value / len_iter   

    return scores