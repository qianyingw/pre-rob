#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 16:36:29 2019

@author: qwang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerNet(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, num_heads, num_encoder_layers, output_dim, pad_idx):
        
        super(TransformerNet, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model = embedding_dim,
                                                   nhead = num_heads, 
                                                   dim_feedforward = 2048, dropout = 0.1)        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers = num_encoder_layers)     
        
        self.linear = nn.Linear(embedding_dim, output_dim) 
     
        
    def forward(self, text):
        """
        Input:
            text: torch tensor, [seq_len, batch_size]            
        Output:
            out: torch tensor, [batch_size, output_dim]       
            
        """  
        embed = self.embedding(text)  # [seq_len, batch_size, embeddgin_dim]
        enc = self.transformer_encoder(embed)  # [seq_len, batch_size, embedding_dim]
        enc = enc.permute(1,0,2)  # [batch_size, seq_len, embedding_dim]
        
        # Obtain encoder output of the last word
        batch_len = enc.size()[0]
        enc_list = []
        for i in batch_len:
            # Append encoder output of last word from each batch           
            # out[i][-1]: [embedding_dim]
            enc_list.append(enc[i][-1])  
        enc_last = torch.stack(enc_list)  # [batch_size, embedding_dim]
        
        z = self.linear(enc_last)  # [batch_size, output_dim]
        z = F.softmax(z, dim=1)  # [batch_size, output_dim]
        
        return z
    
        