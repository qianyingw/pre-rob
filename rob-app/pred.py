#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 16:16:31 2020

@author: qwang
"""

import json
import pandas as pd
import dill
import re

import spacy
nlp = spacy.load("en_core_web_sm")

import torch

from model import ConvNet, AttnNet
from model_han import HAN


import transformers
from transformers import BertConfig, BertTokenizer
from model_bert import BertPoolConv


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
    model.load_state_dict(state_dict)
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
# arg_path = 'pth/bert_w0.json'
# wgt_path = 'pth/biobert'
# pth_path = 'pth/biobert/bert_w0.pth.tar'
# device = torch.device('cpu')

# with open('sample/Minwoo A, 2015.txt', 'r', encoding='utf-8', errors='ignore') as fin:
#     doc = fin.read() 

def pred_prob_bert(arg_path, wgt_path, pth_path, doc, device=torch.device('cpu')):
    # Load args
    with open(arg_path) as f:
        args = json.load(f)['args']
    
    args['wgts_dir'] = wgt_path
    # Load model
    # Tokenizer & Config & Model
    if args['net_type'] == "bert_pool_conv":
        # Tokenizer
        tokenizer = BertTokenizer.from_pretrained(args['wgts_dir'], do_lower_case=True)  
        # Config
        config = BertConfig.from_pretrained(args['wgts_dir'])  
        config.output_hidden_states = True
        config.num_labels = args['num_labels']
        config.unfreeze = args['unfreeze']
        config.pool_method = args['pool_method']
        config.pool_layers = args['pool_layers']
          
        if args['num_hidden_layers']:  config.num_hidden_layers = args['num_hidden_layers']
        if args['num_attention_heads']:  config.num_attention_heads = args['num_attention_heads']
        
        config.num_filters = args['num_filters']
        sizes = args['filter_sizes'].split(',')
        config.filter_sizes = [int(s) for s in sizes]
        model = BertPoolConv.from_pretrained(args['wgts_dir'], config=config)
    
    # Load checkpoint
    checkpoint = torch.load(pth_path, map_location=device)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model.cpu()
     

    # Tokenization
    if type(tokenizer) == transformers.tokenization_bert.BertTokenizer:
        # Convert text to tokens by WordPiece
        tokens = tokenizer.tokenize(doc)
        
        # Split tokens into chunks
        n_chunks = len(tokens) // (args['max_chunk_len'] - 2)
        if len(tokens) % (args['max_chunk_len'] - 2) != 0:
            n_chunks += 1  
                
        # Limit number of chunks
        if n_chunks > args['max_n_chunk']:
            tokens = tokens[:args['max_chunk_len']*args['max_n_chunk']]
            n_chunks = args['max_n_chunk']  

        # Document tensor
        doc_tensor = torch.zeros((n_chunks, 3, args['max_chunk_len']), dtype=torch.long)
        for i in range(n_chunks):
            chunk_tokens = tokens[(args['max_chunk_len']-2) * i : (args['max_chunk_len']-2) * (i+1)]
            chunk_tokens.insert(0, "[CLS]")
            chunk_tokens.append("[SEP]")              
            chunk_tokens_ids = tokenizer.convert_tokens_to_ids(chunk_tokens)
                  
            attn_masks = [1] * len(chunk_tokens_ids)         
            # Pad the last chunk
            while len(chunk_tokens_ids) < args['max_chunk_len']:
                chunk_tokens_ids.append(0)
                attn_masks.append(0)
                
            token_type_ids = [0] * args['max_chunk_len']     
            assert len(chunk_tokens_ids) == args['max_chunk_len'] and len(attn_masks) == args['max_chunk_len']
                         
            doc_tensor[i] = torch.cat((torch.LongTensor(chunk_tokens_ids).unsqueeze(0),
                                    torch.LongTensor(attn_masks).unsqueeze(0),
                                    torch.LongTensor(token_type_ids).unsqueeze(0)), dim=0)
      
    # Prediction
    model.eval()
    doc_tensor = doc_tensor.to(device)
    doc_tensor = doc_tensor.unsqueeze(0)  # bec BertPoolConv input shape is [batch_size, n_chunks, 3, max_chunk_len]
    probs = model(doc_tensor)
    probs = probs.data.cpu().numpy()[0]
    # print("Prob of RoB reported: {:.4f}".format(probs[1]))    
    return probs[1]

#%%
def extract_words(arg_path, field_path, pth_path, doc, num_words, device=torch.device('cpu')):  
    # Load args
    with open(arg_path) as f:
        args = json.load(f)['args']
    
    # Load TEXT field
    with open(field_path, "rb") as fin:
        TEXT = dill.load(fin)   
     
    unk_idx = TEXT.vocab.stoi[TEXT.unk_token]  # 0
    pad_idx = TEXT.vocab.stoi[TEXT.pad_token]  # 1

    # Load model
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
                        output_attn = True)
    
    # Load checkpoint
    checkpoint = torch.load(pth_path, map_location=device)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model.cpu()
     
    # Load pre-trained embedding
    pretrained_embeddings = TEXT.vocab.vectors    
    model.embedding.weight.data.copy_(pretrained_embeddings)
    model.embedding.weight.data[unk_idx] = torch.zeros(args['embed_dim'])  # Zero the initial weights for <unk> tokens
    model.embedding.weight.data[pad_idx] = torch.zeros(args['embed_dim'])  # Zero the initial weights for <pad> tokens
    
    
    # Tokenization
    tokens_uncase = [tok.text for tok in nlp.tokenizer(doc)]
    tokens = [tok.text.lower() for tok in nlp.tokenizer(doc)]  
    idx = [TEXT.vocab.stoi[t] for t in tokens]     
    
    # Prediction
    model.eval()
    doc_tensor = torch.LongTensor(idx).to(device)
    doc_tensor = doc_tensor.unsqueeze(1)  # bec AttnNet input shape is [seq_len, batch_size] 
      
    probs, attn_score = model(doc_tensor)
    attn_score = attn_score.data.cpu().numpy()[0]
    attn_list = list(attn_score.flat)
    
    df = pd.DataFrame({'words': tokens_uncase, 'attn': attn_list})
    df = df.sort_values(by=['attn'], ascending=False)
    
    out = df.head(num_words)
    out = out.reset_index()
    # out_words = list(df['words'][:num_words])
    
    return out #out_words
       
    
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
    model.load_state_dict(state_dict)
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
    




