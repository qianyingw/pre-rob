#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Mon Oct  7 12:13:59 2019
@author: qwang
"""

#%% Setting and Parser
import os
import random
import argparse
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

import utils
from data_helper import DataHelper
import model.net as net


parser = argparse.ArgumentParser()
parser.add_argument('--wdir', default='/home/qwang/rob', help='Project working directory')
parser.add_argument('--data_dir', default='data/psycho', help='Directory containing the dataset')
parser.add_argument('--data_name', default='rob_psycho_fulltokens.json', help='Name of json data file')
parser.add_argument('--model_dir', default='src/model', help='Directory containing params.json')
parser.add_argument('--embed_dir', default='wordvec', help='Directory containing embeddings')
parser.add_argument('--embed_name', default='wikipedia-pubmed-and-PMC-w2v.txt', help='Name of embedding txt file')
parser.add_argument('--rob_item', default='RandomizationTreatmentControl', help='String label of the risk of bis item')
parser.add_argument('--restore_file', default=None, 
                    help='[Optional] Name of the file in params_dir containing weights to reload before training')

args = parser.parse_args()
os.chdir(args.wdir)




###
def train(model, iterator, criterion, optimizer, metrics):
    
    scores = {'loss': 0, 'accuracy': 0, 'f1': 0, 'recall': 0, 'precision': 0, 'specificity': 0}
    len_iter = len(iterator)
    
    model.train()
    
    with tqdm(total=len_iter) as progress_bar:
        for batch in iterator:
            optimizer.zero_grad()
            preds = model(batch.text)
            
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




def train_evaluate(model, train_iterator, valid_iterator, criterion, optimizer, metrics, params, model_dir, restore_file=None):
    """
    
    """
    if restore_file is not None:
        restore_path = os.path.join(model_dir, restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}...".format(restore_path))  
        utils.load_checkpoint(restore_path, model, optimizer)
        
        
    best_valid_loss = float('inf')
    
    for epoch in range(params.n_epochs):
        
        logging.info("\nEpoch {}/{}...".format(epoch+1, params.n_epochs))
     
        train_scores = train(model, train_iterator, criterion, optimizer, metrics)
        valid_scores = evaluate(model, valid_iterator, criterion, metrics)
        
        
        # Save weights if is_best
        is_best = valid_scores['loss'] < best_valid_loss
        utils.save_checkpoint({'epoch': epoch+1,
                               'state_dict': model.state_dict(),
                               'optim_Dict': optimizer.state_dict()},
                              is_best = is_best,
                              checkdir = model_dir)
        
        if is_best:
            logging.info("\n ...Found new lowest loss...")
            best_valid_loss = valid_scores['loss']
            
            # Save the best valid scores in model_dir
            best_loss_path = os.path.join(model_dir, 'best_val_scores.json')
            utils.save_dict_to_json(valid_scores, best_loss_path)
        
        # Save the latest valid scores in model_dir
        last_loss_path = os.path.join(model_dir, 'last_val_scores.json')
        utils.save_dict_to_json(valid_scores, last_loss_path)
        
#        if valid_scores['loss'] < best_valid_loss:
#            best_valid_loss = valid_scores['loss']
#            torch.save(model.state_dict(), 'src/model/model.pt')
        
        print('\n[Train] loss: {0:.3f} | acc: {1:.2f}% | f1: {2:.2f}% | recall: {3:.2f}% | precision: {4:.2f}% | specificity: {5:.2f}%'.format(
            train_scores['loss'], train_scores['accuracy']*100, train_scores['f1']*100, train_scores['recall']*100, train_scores['precision']*100, train_scores['specificity']*100))
        print('[Val] loss: {0:.3f} | acc: {1:.2f}% | f1: {2:.2f}% | recall: {3:.2f}% | precision: {4:.2f}% | specificity: {5:.2f}%'.format(
            valid_scores['loss'], valid_scores['accuracy']*100, valid_scores['f1']*100, valid_scores['recall']*100, valid_scores['precision']*100, valid_scores['specificity']*100))
        
        
def test(model, test_iterator, criterion, metrics, model_dir, restore_file):   
     
    utils.load_checkpoint(os.path.join(model_dir, restore_file + '.pth.tar'), model)
    test_scores = evaluate(model, test_iterator, criterion, metrics)
    save_path = os.path.join(model_dir, "test_scores.json")
    utils.save_dict_to_json(test_scores, save_path)
    
    print('\n[Test] loss: {0:.3f} | acc: {1:.2f}% | f1: {2:.2f}% | recall: {3:.2f}% | precision: {4:.2f}% | specificity: {5:.2f}%'.format(
            test_scores['loss'], test_scores['accuracy']*100, test_scores['f1']*100, test_scores['recall']*100, test_scores['precision']*100, test_scores['specificity']*100))



#%% Main
if __name__ == '__main__':
    
    # Load parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No params configuration found at {}".format(json_path)
    params = utils.Params(json_path)
    
    # Modify and save params.json
    params.cuda = torch.cuda.is_available()
    params.n_epochs = 1
    params.batch_size = 128
    params.save(json_path)
    params.update(json_path)
    
    # Check GPU
    device = torch.device('cuda' if params.cuda else 'cpu')
    
    # Set random seed
    random.seed(params.seed)
    torch.manual_seed(params.seed)
    if device == 'cuda': torch.cuda.manual_seed(params.seed)
    torch.backends.cudnn.deterministic = True  
    
    # Set logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))
    
    # Load data and create iterators
    logging.info("Loading the datasets...")
    helper = DataHelper(data_dir = args.data_dir, data_name = args.data_name,
                        params_dir = args.model_dir,
                        embed_dir = args.embed_dir, embed_name = args.embed_name, 
                        params = params)
    train_data, valid_data, test_data = helper.create_data(rob_item = args.rob_item)   
    train_iterator, valid_iterator, test_iterator = helper.create_iterators(train_data, valid_data, test_data)
    logging.info("Done.")

    # Define the model
    input_dim = len(helper.TEXT.vocab)
    output_dim = len(helper.LABEL.vocab)
    
    unk_idx = helper.TEXT.vocab.stoi[helper.TEXT.unk_token]
    pad_idx = helper.TEXT.vocab.stoi[helper.TEXT.pad_token]
    
    model = net.CNN(vocab_size = input_dim,
                    embedding_dim = params.embedding_dim, 
                    n_filters = params.n_filters, 
                    filter_sizes = params.filter_sizes, 
                    output_dim = output_dim, 
                    dropout = params.dropout, 
                    pad_idx = pad_idx)
    
    print(model)
    
    # Load pre-trained embedding
    pretrained_embeddings = helper.TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)
    # Zero the initial weights of the unknown and padding tokens
    model.embedding.weight.data[unk_idx] = torch.zeros(params.embedding_dim)
    model.embedding.weight.data[pad_idx] = torch.zeros(params.embedding_dim)
    
    # Define the optimizer, loss function and metrics
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()    
    metrics = net.metrics
    
    model = model.to(device)
    criterion = criterion.to(device)
    
    # Train the model
    logging.info("\nStart training for {} epoch(s)...".format(params.n_epochs)) 
    train_evaluate(model, train_iterator, valid_iterator, criterion, optimizer, metrics, params, args.model_dir, args.restore_file)
    
    # Test
    logging.info("\nStart testing...")
    test(model, test_iterator, criterion, metrics, args.model_dir, restore_file = 'best')

