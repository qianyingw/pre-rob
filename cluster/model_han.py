#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 14:40:34 2019

@author: qwang
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class WordAttn(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, word_hidden_dim, word_num_layers, dropout, pad_idx):
        
        super(WordAttn, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.gru = nn.GRU(input_size = embedding_dim, hidden_size = word_hidden_dim,
                          num_layers = word_num_layers, batch_first=True, 
                          dropout = dropout, bidirectional = True)
        
        self.tanh = nn.Tanh()        
        
        # Initialize weight
        w = torch.empty(2*word_hidden_dim, 2*word_hidden_dim)
        nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
        self.w = nn.Parameter(w)
        # Initialize bias
        b = torch.zeros(2*word_hidden_dim) 
        self.b = nn.Parameter(b)      
        # Initialize word context vector c_w
        c_w = torch.empty(2*word_hidden_dim, 1)
        nn.init.kaiming_uniform_(c_w, mode='fan_in', nonlinearity='relu')
        self.c_w = nn.Parameter(c_w)
        
    
    def forward(self, sent):
        """
            Input
                sent: [sent_len, batch_size]
            Output
                s_i: [batch_size, sent_len, 2*hidden_size]
        """
        embed = self.embedding(sent)  # [sent_len, batch_size, embedding_dim]
        embed = embed.permute(1,0,2)  # [batch_size, sent_len, embedding_dim]
        
        h_i, h_n = self.gru(embed)  # h_i: [batch_size, sent_len, 2*word_hidden_dim]
                
        u_i = self.tanh(torch.matmul(h_i, self.w) + self.b)  # [batch_size, sent_len, 2*word_hidden_dim]      
        a_i = F.softmax(torch.matmul(u_i, self.c_w), dim=1)  # [batch_size, sent_len, 1]
        
        # Obtain sentence vector s_i
        s_i = torch.matmul(h_i.permute(0,2,1), a_i)  # [batch_size, 2*word_hidden_dim, 1]
        s_i = s_i.squeeze(2)  # [batch_size, 2*word_hidden_dim]
        
        return s_i



class SentAttn(nn.Module):
    def __init__(self, word_hidden_dim, sent_hidden_dim, sent_num_layers, dropout, output_dim):
        
        super(SentAttn, self).__init__()
        
        
        self.gru = nn.GRU(input_size = 2*word_hidden_dim, hidden_size = sent_hidden_dim,
                          num_layers = sent_num_layers, batch_first=True, 
                          dropout = dropout, bidirectional = True)
        
        self.tanh = nn.Tanh()        
        self.linear = nn.Linear(2*sent_hidden_dim, output_dim)
        
        # Initialize weight
        w = torch.empty(2*sent_hidden_dim, 2*sent_hidden_dim)
        nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
        self.w = nn.Parameter(w)
        
        # Initialize bias
        b = torch.zeros(2*sent_hidden_dim) 
        self.b = nn.Parameter(b) 
        
        # Initialize sentence context vector c_s
        c_s = torch.empty(2*sent_hidden_dim, 1)
        nn.init.kaiming_uniform_(c_s, mode='fan_in', nonlinearity='relu')
        self.c_s = nn.Parameter(c_s)
        
    def forward(self, sent_mat):
        """
            Input
                sent_mat: [batch_size, num_sents, 2*word_hidden_dim]
            Output
                z: [batch_size, output_dim]
        """
        h, h_n = self.gru(sent_mat)  # h: [batch_size, num_sents, 2*sent_hidden_dim]
        
        u = self.tanh(torch.matmul(h, self.w) + self.b)  # [batch_size, num_sents, 2*sent_hidden_dim]  
        a = F.softmax(torch.matmul(u, self.c_s), dim=1)  # [batch_size, num_sents, 1]
        
        # Obtain document vector d
        d = torch.matmul(h.permute(0,2,1), a)  # [batch_size, 2*sent_hidden_dim, 1]
        d = d.squeeze(2)  # [batch_size, 2*sent_hidden_dim]
        
        z = F.softmax(d, dim=1)  # [batch_size, 2*sent_hidden_dim]
        z = self.linear(z)  # [batch_size, output_dim]
        
        return z


class HAN(nn.Module):
    
    def __init__(self, ):
        
        super(HAN, self).__init__()
        
        self.word_attn = WordAttn()
        self.sent_attn = SentAttn()
    