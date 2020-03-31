#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 14:40:34 2019

@author: qwang
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

#%%
class WordAttn(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, word_hidden_dim, word_num_layers, pad_idx):
        
        super(WordAttn, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.gru = nn.GRU(input_size = embedding_dim, hidden_size = word_hidden_dim,
                          num_layers = word_num_layers, batch_first=True, bidirectional = True)
        
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
                sent: [batch_size, sent_len]
            Output
                a_i: [batch_size, sent_len]
                s_i: [batch_size, 2*word_hidden_dim]
        """
        embed = self.embedding(sent)  # [batch_size, sent_len, embedding_dim]
        
        h_i, h_n = self.gru(embed)  # h_i: [batch_size, sent_len, 2*word_hidden_dim]
                
        u_i = self.tanh(torch.matmul(h_i, self.w) + self.b)  # [batch_size, sent_len, 2*word_hidden_dim]      
        a_i = F.softmax(torch.matmul(u_i, self.c_w), dim=1)  # [batch_size, sent_len, 1]
        
        # Obtain sentence vector s_i
        s_i = torch.matmul(h_i.permute(0,2,1), a_i)  # [batch_size, 2*word_hidden_dim, 1]
        
        a_i = a_i.squeeze(2)  # [batch_size, sent_len]
        s_i = s_i.squeeze(2)  # [batch_size, 2*word_hidden_dim]
        
        
        return a_i, s_i



#%%
class SentAttn(nn.Module):
    def __init__(self, word_hidden_dim, sent_hidden_dim, sent_num_layers):
        
        super(SentAttn, self).__init__()
               
        self.gru = nn.GRU(input_size = 2*word_hidden_dim, hidden_size = sent_hidden_dim,
                          num_layers = sent_num_layers, batch_first=True, bidirectional = True)
        
        self.tanh = nn.Tanh()        
        
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
        
    def forward(self, doc):
        """
            Input
                doc: [batch_size, num_sents, 2*word_hidden_dim]
            Output
                a: [batch_size, num_sents]
                d: [batch_size, 2*sent_hidden_dim]
                # z: [batch_size, output_dim]
        """
        h, h_n = self.gru(doc)  # h: [batch_size, num_sents, 2*sent_hidden_dim]
        
        u = self.tanh(torch.matmul(h, self.w) + self.b)  # [batch_size, num_sents, 2*sent_hidden_dim]  
        a = F.softmax(torch.matmul(u, self.c_s), dim=1)  # [batch_size, num_sents, 1]
        
        
        # Obtain document vector d
        d = torch.matmul(h.permute(0,2,1), a)  # [batch_size, 2*sent_hidden_dim, 1]
        a = a.squeeze(2) # [batch_size, num_sents]
        d = d.squeeze(2)  # [batch_size, 2*sent_hidden_dim]
        
#        d = F.softmax(d, dim=1)  # [batch_size, 2*sent_hidden_dim]
#        z = self.linear(z)  # [batch_size, output_dim]
        
        return a, d

#%%
class HAN(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, word_hidden_dim, word_num_layers, pad_idx,
                       sent_hidden_dim, sent_num_layers, output_dim):
        
        super(HAN, self).__init__()
        
        self.word_attn = WordAttn(vocab_size = vocab_size,
                                  embedding_dim = embedding_dim,
                                  word_hidden_dim = word_hidden_dim,
                                  word_num_layers = word_num_layers,
                                  pad_idx = pad_idx)
        
        self.sent_attn = SentAttn(word_hidden_dim = word_hidden_dim,
                                  sent_hidden_dim = sent_hidden_dim,
                                  sent_num_layers = sent_num_layers)
        
        self.linear = nn.Linear(2*sent_hidden_dim, output_dim)
    
    
    def forward(self, text):
        """
            Input
                text: [batch_size, max_doc_len, max_sent_len]
            Output
                z: [batch_size, output_dim]
        """
        text = text.permute(1, 0, 2)  # [max_doc_len, batch_size, max_sent_len]
        word_a_ls, word_s_ls = [], []
        
        for sent in text:
            # sent: [batch_size, max_sent_len]
            word_a, word_s = self.word_attn(sent)
            word_a = word_a.unsqueeze(1)  # word_a: [batch_size, 1, max_sent_len]
            word_s = word_s.unsqueeze(1)  # word_s: [batch_size, 1, 2*word_hidden_dim]
                       
            word_a_ls.append(word_a)  # [batch_size, 1, max_sent_len] * max_doc_len
            word_s_ls.append(word_s)  # [batch_size, 1, 2*word_hidden_dim] * max_doc_len
        
        self.sent_a = torch.cat(word_a_ls, 1)  # sent_a: [batch_size, max_doc_len, max_sent_len]  
        sent_s = torch.cat(word_s_ls, 1)       # sent_s: [batch_size, max_doc_len, 2*word_hidden_dim]
        
        # doc_a: [batch_size, max_doc_len]
        # doc_s: [batch_size, 2*sent_hidden_dim]
        self.doc_a, doc_s = self.sent_attn(sent_s)
        
        z = self.linear(doc_s)  # [batch_size, output_dim]
        z = F.softmax(z, dim=1)
        
        return z


#%% Instance
#batch_size = 20
#vocab_size = 5000
#embed_dim = 200
#max_sent_len = 15
#max_doc_len = 8
#output_dim = 2
#word_hidden_dim = 22
#sent_hidden_dim = 33
#word_num_layers = 1
#sent_num_layers = 1
#pad_idx = 1
#
#word_attn = WordAttn(vocab_size, embed_dim, word_hidden_dim, word_num_layers, pad_idx)
#sent_attn = SentAttn(word_hidden_dim, sent_hidden_dim, sent_num_layers)
#linear = nn.Linear(2*sent_hidden_dim, output_dim)
#
#
#
#X = torch.randint(999, (batch_size, max_doc_len, max_sent_len))
#X = X.permute(1, 0, 2)  # [max_doc_len, batch_size, max_sent_len]
#X.shape
#
#word_a_ls, word_s_ls = [], []      
#for sent in X:
#    word_a, word_s = word_attn(sent)  # sent: [batch_size, max_sent_len]
#    word_a = word_a.unsqueeze(1)  # word_a: [batch_size, 1, max_sent_len]
#    word_s = word_s.unsqueeze(1)  # word_s: [batch_size, 1, 2*word_hidden_dim]               
#    word_a_ls.append(word_a)  # [batch_size, 1, max_sent_len] * max_doc_len
#    word_s_ls.append(word_s)  # [batch_size, 1, 2*word_hidden_dim] * max_doc_len
#
#sent_a = torch.cat(word_a_ls, 1)  # sent_a: [batch_size, max_doc_len, max_sent_len]  
#sent_s = torch.cat(word_s_ls, 1)  # sent_s: [batch_size, max_doc_len, 2*word_hidden_dim]
#doc_a, doc_s = sent_attn(sent_s)
#z = linear(doc_s)  # [batch_size, output_dim]
#z = F.softmax(z, dim=1)
#
#print(sent_a.shape)
#print(sent_s.shape)
#print(doc_a.shape)
#print(doc_s.shape)
#print(z.shape)
