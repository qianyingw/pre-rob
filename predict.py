#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 16:33:29 2019

@author: qwang
"""

import data_process.tokenizer as tokenizer
import utils
from data_helper import DataHelper
import torch

device = torch.device("cuda" if torch.cuda.is_available else "cpu")



params = utils.Params(json_path = 'src/model/params.json')


helper = DataHelper(data_dir = 'data/psycho', data_name = 'rob_psycho_fulltokens.json',
                    params_dir = 'src/model',
                    embed_dir = 'wordvec', embed_name = 'wikipedia-pubmed-and-PMC-w2v.txt', 
                    params = params)

train_data, valid_data, test_data = helper.create_data(rob_item = 'RandomizationTreatmentControl')


helper.TEXT.vocab.itos[:5]  # ['<unk>', '<pad>', ',', '.', 'the']

len(helper.TEXT.vocab)  
len(helper.LABEL.vocab)
len(helper.ID.vocab)  # 2406

helper.TEXT.pad_token  # '<pad>'
helper.TEXT.unk_token  # '<unk>'
helper.TEXT.vocab.stoi[helper.TEXT.pad_token]  # 1
helper.TEXT.vocab.stoi[helper.TEXT.unk_token]  # 0


def predict_doc(model, text, min_len = 200):
    model.eval()
    text = tokenizer.preprocess_text(text)
    tokens = tokenizer.tokenize_text(text)
    if len(tokens) < min_len:
        tokens = tokens + ['<pad>'] * (min_len - len(tokens))
    idxs = [helper.TEXT.vocab.stoi[t] for t in tokens]
    
    tensor = torch.LongTensor(idxs).to(device)
    tensor = tensor.unsqueeze(1)
    preds = torch.sigmoid(model(tensor))
    return preds.item()
    