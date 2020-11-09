#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 20:44:52 2020

@author: qwang
"""

import json
import pandas as pd
import dill
import re

import spacy
nlp = spacy.load("en_core_web_sm")

import torch

from transformers import DistilBertTokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
from sentence_transformers import SentenceTransformer, util
sent_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')


from model import ConvNet, AttnNet, HAN, DistilClsConv

#%%
def pred_prob(arg_path, field_path, pth_path, doc, device=torch.device('cpu')):
    
    # Load args
    with open(arg_path) as f:
        args = json.load(f)['args']
    
    # Load TEXT field
    with open(field_path,"rb") as fin:
        TEXT = dill.load(fin)   
     
    unk_idx = TEXT.vocab.stoi[TEXT.unk_token]  # 0
    pad_idx = TEXT.vocab.stoi[TEXT.pad_token]  # 1

    # Load model
    if args['net_type'] == 'cnn':
        sizes = args['filter_sizes'].split(',')
        sizes = [int(s) for s in sizes]
        model = ConvNet(vocab_size = args['max_vocab_size'] + 2,
                        embedding_dim = args['embed_dim'], 
                        n_filters = args['num_filters'], 
                        filter_sizes = sizes, 
                        output_dim = 2, 
                        dropout = args['dropout'],
                        pad_idx = pad_idx,
                        embed_trainable = args['embed_trainable'],
                        batch_norm = args['batch_norm'])
    
    if args['net_type'] == 'attn':       
        model = AttnNet(vocab_size = args['max_vocab_size'] + 2, 
                        embedding_dim = args['embed_dim'], 
                        rnn_hidden_dim = args['rnn_hidden_dim'], 
                        rnn_num_layers = args['rnn_num_layers'], 
                        output_dim = 2, 
                        bidirection = args['bidirection'], 
                        rnn_cell_type = args['rnn_cell_type'], 
                        dropout = args['dropout'], 
                        pad_idx = pad_idx,
                        embed_trainable = args['embed_trainable'],
                        batch_norm = args['batch_norm'],
                        output_attn = False)
    
    # Load checkpoint
    checkpoint = torch.load(pth_path, map_location=device)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict, strict=False)
    model.cpu()
     
    # Load pre-trained embedding
    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)
    model.embedding.weight.data[unk_idx] = torch.zeros(args['embed_dim'])  # Zero the initial weights for <unk> tokens
    model.embedding.weight.data[pad_idx] = torch.zeros(args['embed_dim'])  # Zero the initial weights for <pad> tokens
    
    # Tokenization
    tokens = [tok.text.lower() for tok in nlp.tokenizer(doc)]  
    idx = [TEXT.vocab.stoi[t] for t in tokens]     
    
    
    while len(idx) < args['max_token_len']:
        idx = idx + [1]*args['max_token_len']
  
    if len(idx) > args['max_token_len']:
        idx = idx[:args['max_token_len']]
    
    # Prediction
    model.eval()
    doc_tensor = torch.LongTensor(idx).to(device)
    doc_tensor = doc_tensor.unsqueeze(1)  # bec AttnNet input shape is [seq_len, batch_size] 
    probs = model(doc_tensor)
    probs = probs.data.cpu().numpy()[0]
    # print("Prob of RoB reported: {:.4f}".format(probs[1]))
    
    return probs[1]


#%%
# arg_path = 'pth/dsc_w0.json'
# pth_path = 'pth/dsc_w0.pth.tar'
# device = torch.device('cpu')
# with open('sample/Minwoo A, 2015.txt', 'r', encoding='utf-8', errors='ignore') as fin:
#     text = fin.read() 

def pred_prob_distil(arg_path, pth_path, text, max_n_sent=30, device=torch.device('cpu')):
    # Load args
    with open(arg_path) as f:
        args = json.load(f)['args']
        
    if args['rob_sent'] is None:
        if args['rob_item'] == 'RandomizationTreatmentControl':
            rob_sent = 'Animals are randomly allocated to treatment or control groups at the start of the experimental treatment'
        if args['rob_item'] == 'BlindedOutcomeAssessment':
            rob_sent = 'Assessment of an outcome in a blinded fashion. Investigators measuring the outcome do not know which treatment group the animals belongs to and what treatment they had received' 
        if args['rob_item'] == 'SampleSizeCalculation':
            rob_sent = 'The manuscript reports the performance of a sample size calculation and describes how this number was derived statistically' 
        if args['rob_item'] == 'AnimalExclusions':
            rob_sent = 'All animals, all data and all outcomes measured are accounted for and presented in the final analysis. Reasons are given for animal exclusions' 
        if args['rob_item'] == 'AllocationConcealment':
            rob_sent = 'Investigators performing the experiment do not know which treatment an animal is being given' 
        if args['rob_item'] == 'AnimalWelfareRegulations':
            rob_sent = 'Research investigators complied with animal welfare regulations' 
        if args['rob_item'] == 'ConflictsOfInterest':
            rob_sent = 'Potential conflict of interest, like funding or affiliation to a pharmaceutical company' 
            
    ### Load model ###
    model = DistilClsConv.from_pretrained('distilbert-base-uncased', return_dict=True)

    ### Load checkpoint ###
    checkpoint = torch.load(pth_path, map_location=device)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict, strict=False)
    model.cpu()
     
    ### Tokenization ###
    doc = nlp(text)
    # Convert spacy span to string list 
    sents = list(doc.sents)  
    sents = [str(s) for s in sents]     
    # Remove too short sentences
    sents = [s for s in sents if len(s.split(' ')) > 3]
    
    # Compute cosine-similarities for rob sentence with each sentence in fulltext
    sent_embeds = sent_model.encode(sents, convert_to_tensor=True)
    rob_embed = sent_model.encode([rob_sent], convert_to_tensor=True)
    
    cos = util.pytorch_cos_sim(sent_embeds, rob_embed)
    pairs = []
    for i in range(cos.shape[0]):
        pairs.append({'index': i, 'score': cos[i][0]})   
    pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)  # sort scores in decreasing order
            
    sim_sents = []
    for pair in pairs[:max_n_sent]:
        sim_sents.append(sents[pair['index']])
    
    sim_text = ". ".join(sim_sents)            
    
    # Prediction
    model.eval()
    inputs = tokenizer(sim_text, padding=True, truncation=True, return_tensors="pt")    
    probs = model(**inputs)
    probs = probs.data.cpu().numpy()[0]
    # print("Prob of RoB reported: {:.4f}".format(probs[1]))    
    return probs[1]

       
    
#%%    
def extract_sents(arg_path, field_path, pth_path, doc, num_sents,  device=torch.device('cpu')):  
    # Load args
    with open(arg_path) as f:
        args = json.load(f)['args']
    
    # Load TEXT field
    with open(field_path, "rb") as fin:
        TEXT = dill.load(fin)   
     
    unk_idx = TEXT.vocab.stoi[TEXT.unk_token]  # 0
    pad_idx = TEXT.vocab.stoi[TEXT.pad_token]  # 1

    model = HAN(vocab_size = args['max_vocab_size'] + 2,
                embedding_dim = args['embed_dim'], 
                word_hidden_dim = args['word_hidden_dim'],
                word_num_layers = args['word_num_layers'],
                pad_idx = pad_idx,    
                embed_trainable = args['embed_trainable'],
                batch_norm = args['batch_norm'],
                sent_hidden_dim = args['sent_hidden_dim'],
                sent_num_layers = args['sent_num_layers'],
                output_dim = 2,
                output_attn = True)
    
    # Load checkpoint
    checkpoint = torch.load(pth_path, map_location=device)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict, strict=False)
    model.cpu()
     
    # Load pre-trained embedding
    pretrained_embeddings = TEXT.vocab.vectors    
    model.word_attn.embedding.weight.data.copy_(pretrained_embeddings) 
    model.word_attn.embedding.weight.data[unk_idx] = torch.zeros(args['embed_dim'])  # Zero the initial weights for <unk> tokens
    model.word_attn.embedding.weight.data[pad_idx] = torch.zeros(args['embed_dim'])  # Zero the initial weights for <pad> tokens

        
    # Split document into sentencces   
    doc = nlp(doc)
    sents = [sent.text for sent in doc.sents]
      
    sent_token = []
    sent_idx = []
    
    for sent in sents:            
        tokens = [token.text for token in nlp(sent)]
        sent_token.append(tokens) 
    sent_token = [st for st in sent_token if len(st) > 10]

 
    for tokens in sent_token:            
        idx = [TEXT.vocab.stoi[t.lower()] for t in tokens]  
        if len(idx) < args['max_sent_len']:
            idx = idx + [1]*(args['max_sent_len']-len(idx))  # pad sent
        else:
            idx = idx[:args['max_sent_len']]  # cut sent
        sent_idx.append(idx)
        
        
    while len(sent_idx) < args['max_doc_len']:
        sent_idx.append([1]*args['max_sent_len'])
       
    if len(sent_idx) > args['max_doc_len']:
        sent_idx = sent_idx[:args['max_doc_len']]
    
    # Prediction
    model.eval()
    doc_tensor = torch.LongTensor(sent_idx).to(device)
    doc_tensor = doc_tensor.unsqueeze(0)  # bec HAN input shape is [batch_size, max_doc_len, max_sent_len] 
      
    probs, attn_score = model(doc_tensor)
    attn_score = attn_score.data.cpu().numpy()[0]
    attn_list = list(attn_score.flat)
    
    if len(sent_token) > len(attn_list):
        df = pd.DataFrame({'sent_token': sent_token[:len(attn_list)], 'attn': attn_list[:len(sent_token)]})
    else:     
        df = pd.DataFrame({'sent_token': sent_token, 'attn': attn_list[:len(sent_token)]})
    df = df.sort_values(by=['attn'], ascending=False)
    
    out_sent_tokens = list(df['sent_token'][:num_sents])
    # out_sents = [' '.join(sent) for sent in out_sent_tokens]
    
    out_sents = []
    for s in out_sent_tokens:
        sent = ' '.join(s)
        sent = re.sub(r" \.", ".", sent)
        sent = re.sub(r" \,", ",", sent)
        try:
            sent = re.sub(r"\( ", "(", sent)
            sent = re.sub(r" \)", ")", sent)
            sent = re.sub(r"\[ ", "[", sent)
            sent = re.sub(r" \]", "]", sent)
            sent = re.sub(r" \- ", "-", sent)
        except:
            pass
        out_sents.append(sent)
        
    return out_sents  
    




