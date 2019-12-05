#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 17:28:44 2019
github.com/CSTR-Edinburgh/mlpractical/blob/mlp2019-20/mlp_cluster_tutorial/experiment_builder.py
@author: qwang

"""

import os
import logging
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np

import utils
import json


# from utils import save_statistics
      

def train(model, iterator, criterion, optimizer, metrics):
    
    scores = {'loss': 0, 'accuracy': 0, 'f1': 0, 'recall': 0, 'precision': 0, 'specificity': 0}
    len_iter = len(iterator)
    
    model.train()
    
    with tqdm(total=len_iter) as progress_bar:
        for batch in iterator:
            optimizer.zero_grad()
            preds = model(batch.text)  # preds.size() = [batch_size, output_dim]
            
            loss = criterion(preds, batch.label)       
            epoch_scores = metrics(preds, batch.label)  # dictionary of 5 metric scores
                   
            loss.backward()
            optimizer.step()
            
            scores['loss'] += loss.item()
            for key, value in epoch_scores.items():               
                scores[key] += value        
            progress_bar.update(1)  # update progress bar 
    
    for key, value in scores.items():
        scores[key] = value / len_iter
    
    return scores


def evaluate(model, iterator, criterion, metrics):
    
    scores = {'loss': 0, 'accuracy': 0, 'f1': 0, 'recall': 0, 'precision': 0, 'specificity': 0}
    len_iter = len(iterator)
    
    model.eval()
    
    with torch.no_grad():
        with tqdm(total=len_iter) as progress_bar:
            for batch in iterator:
                preds = model(batch.text)
                
                loss = criterion(preds, batch.label)
                epoch_scores = metrics(preds, batch.label)  # # dictionary of 5 metric scores
                
                scores['loss'] += loss.item()
                for key, value in epoch_scores.items():               
                    scores[key] += value                        
                progress_bar.update(1)
                        
    for key, value in scores.items():
        scores[key] = value / len_iter
    
    return scores



def train_evaluate(model, train_iterator, valid_iterator, criterion, optimizer, metrics, args, exp_dir, restore_file=None):
    """
    
    """
    if restore_file is not None:
        restore_path = os.path.join(exp_dir, restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}...".format(restore_path))  
        utils.load_checkpoint(restore_path, model, optimizer)
        
        
    best_valid_loss = float('inf')
    
    
    train_acc_list = []
    train_loss_list = []
    val_acc_list = []
    val_loss_list = []
     
    # Create args and output dict
    output_dict = {'args': vars(args),
                   'prfs': {}}
       
    for epoch in range(args.num_epochs):
     
        logging.info("\nEpoch {}/{}...".format(epoch+1, args.num_epochs))
        
        train_scores = train(model, train_iterator, criterion, optimizer, metrics)
        valid_scores = evaluate(model, valid_iterator, criterion, metrics)        
        
        # Update output dictionary
        output_dict['prfs'][str('train_'+str(epoch+1))] = train_scores
        output_dict['prfs'][str('valid_'+str(epoch+1))] = valid_scores
                           
        # Save weights if is_best
        is_best = valid_scores['loss'] < best_valid_loss
        utils.save_checkpoint({'epoch': epoch+1,
                               'state_dict': model.state_dict(),
                               'optim_Dict': optimizer.state_dict()},
                              is_best = is_best,
                              checkdir = exp_dir)
        
        if is_best:
            # logging.info("\n ...Found new lowest loss...")
            best_valid_loss = valid_scores['loss']
            
            # Save the best valid scores in exp_dir
            best_loss_path = os.path.join(exp_dir, 'best_val_scores.json')
            utils.save_dict_to_json(valid_scores, best_loss_path)
        
        # Save the latest valid scores in exp_dir
        last_loss_path = os.path.join(exp_dir, 'last_val_scores.json')
        utils.save_dict_to_json(valid_scores, last_loss_path)
        
#        if valid_scores['loss'] < best_valid_loss:
#            best_valid_loss = valid_scores['loss']
#            torch.save(model.state_dict(), 'src/model/model.pt')
        
        train_acc_list.append(train_scores['accuracy'])
        train_loss_list.append(train_scores['loss'])
        val_acc_list.append(valid_scores['accuracy'])
        val_loss_list.append(valid_scores['loss'])
        
        print('\nEpochs {}/{}...'.format(epoch+1, args.num_epochs))       
        print('\n[Train] loss: {0:.3f} | acc: {1:.2f}% | f1: {2:.2f}% | recall: {3:.2f}% | precision: {4:.2f}% | specificity: {5:.2f}%'.format(
            train_scores['loss'], train_scores['accuracy']*100, train_scores['f1']*100, train_scores['recall']*100, train_scores['precision']*100, train_scores['specificity']*100))
        print('[Val] loss: {0:.3f} | acc: {1:.2f}% | f1: {2:.2f}% | recall: {3:.2f}% | precision: {4:.2f}% | specificity: {5:.2f}%'.format(
            valid_scores['loss'], valid_scores['accuracy']*100, valid_scores['f1']*100, valid_scores['recall']*100, valid_scores['precision']*100, valid_scores['specificity']*100))
    
    # Write performance to 'expname_prfs.json'
    with open(os.path.join(exp_dir, args.exp_name+'_prfs.json'), 'w') as fout:
        json.dump(output_dict, fout, indent=4)
                   
    return train_acc_list, train_loss_list, val_acc_list, val_loss_list
    
    
        
def test(model, test_iterator, criterion, metrics, exp_dir, restore_file):   
     
    utils.load_checkpoint(os.path.join(exp_dir, restore_file + '.pth.tar'), model)
    test_scores = evaluate(model, test_iterator, criterion, metrics)
    save_path = os.path.join(exp_dir, "test_scores.json")
    utils.save_dict_to_json(test_scores, save_path)
  
    print('\n[Test] loss: {0:.3f} | acc: {1:.2f}% | f1: {2:.2f}% | recall: {3:.2f}% | precision: {4:.2f}% | specificity: {5:.2f}%'.format(
            test_scores['loss'], test_scores['accuracy']*100, test_scores['f1']*100, test_scores['recall']*100, test_scores['precision']*100, test_scores['specificity']*100))
    
    return test_scores


def plot_performance(train_acc_list, train_loss_list, val_acc_list, val_loss_list, png_dir):
    # Figure size
    plt.figure(figsize=(15,5))

    x = np.arange(len(train_loss_list))+1
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.title("Loss")
    plt.plot(x, train_loss_list, label="train")
    plt.plot(x, val_loss_list, label="val")
    plt.legend(loc='upper right')

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    plt.plot(x, train_acc_list, label="train")
    plt.plot(x, val_acc_list, label="val")
    plt.legend(loc='lower right')

    # Save figure
    plt.savefig(os.path.join(png_dir, "performance.png"))

    # Show plots
    # plt.show()
        