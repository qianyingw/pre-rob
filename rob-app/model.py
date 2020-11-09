#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 17:12:41 2019

@author: qwang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import DistilBertModel, DistilBertPreTrainedModel

#%%
class ConvNet(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx, embed_trainable, batch_norm):
    
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        if embed_trainable == False:
            self.embedding.weight.requires_grad = False  # Freeze embedding
            
        self.convs = nn.ModuleList([nn.Conv2d(in_channels = 1,
                                              out_channels = n_filters,
                                              kernel_size = (fsize, embedding_dim)) for fsize in filter_sizes
                                   ])                
        self.fc = nn.Linear(n_filters * len(filter_sizes), output_dim)
        self.fc_bn = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.apply_bn = batch_norm
    
    
    def forward(self, text):
        """
        Params:
            text: torch tensor, [seq_len, batch_size]            
        Yields:
            out: torch tensor, [batch_size, output_dim]       
            
        """         
        embed = self.embedding(text)  # [seq_len, batch_size, embedding_dim]
        embed = embed.permute(1,0,2)  # [batch_size, seq_len, embedding_dim]
        embed = embed.unsqueeze(1)  # [batch_size, 1, seq_len, embedding_dim]
        
        conved = [F.relu(conv(embed)) for conv in self.convs]  # [batch_size, n_filters, (seq_len-fsize+1), 1]
        conved = [conv.squeeze(3) for conv in conved]  # [batch_size, n_filters, (seq_len-fsize+1)]
        pooled = [F.max_pool1d(conv, conv.shape[2]) for conv in conved]  # [batch_size, n_filters, 1]
        pooled = [pool.squeeze(2) for pool in pooled]  # [batch_size, n_filters]
        
        cat = torch.cat(pooled, dim=1)  # [batch_size, n_filters * len(filter_sizes)]
        dp = self.dropout(cat)
        out = self.fc(dp)  # # [batch_size, output_dim]
        if self.apply_bn == True:
            out = self.fc_bn(out)
        out = F.softmax(out, dim=1)  # [batch_size, output_dim]  
        
        return out


#%%
class AttnNet(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, rnn_hidden_dim, rnn_num_layers, output_dim, 
                 bidirection, rnn_cell_type, dropout, pad_idx, 
                 embed_trainable, batch_norm, output_attn):
        
        super(AttnNet, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        if embed_trainable == False:
            self.embedding.weight.requires_grad = False  # Freeze embedding
            
        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size = rnn_hidden_dim,
                            num_layers = rnn_num_layers, dropout = 0, 
                            batch_first = True, bidirectional = bidirection)
        
        self.gru = nn.GRU(input_size = embedding_dim, hidden_size = rnn_hidden_dim,
                          num_layers = rnn_num_layers, dropout = 0, 
                          batch_first = True, bidirectional = bidirection)
        
        num_directions = 2 if bidirection == True else 1
        self.rnn_cell_type = rnn_cell_type
        self.apply_bn = batch_norm
        self.output_attn = output_attn
        
        
        # Initialize weight
        w = torch.empty(num_directions*rnn_hidden_dim, num_directions*rnn_hidden_dim)
        nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
        self.w = nn.Parameter(w)
        # Initialize bias
        b = torch.zeros(num_directions*rnn_hidden_dim) # b = torch.zeros(batch_maxlen, 1)
        self.b = nn.Parameter(b)      
        # Initialize context vector c
        c = torch.empty(num_directions*rnn_hidden_dim, 1)
        nn.init.kaiming_uniform_(c, mode='fan_in', nonlinearity='relu')
        self.c = nn.Parameter(c)
        
        
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        self.linear = nn.Linear(num_directions*rnn_hidden_dim, output_dim)
        self.fc_bn = nn.BatchNorm1d(output_dim)
        
        
    def forward(self, text):
        
        embed = self.embedding(text)  # [seq_len, batch_size, embedding_dim]
        embed = embed.permute(1,0,2)  # [batch_size, seq_len, embedding_dim]
        
        # a: [batch_size, seq_len, num_directions*hidden_dim], output features from last layer for each t
        # h_n: [batch_size, num_layers*num_directions, hidden_dim], hidden state for t=seq_len
        # c_n: [batch_size, num_layers*num*directions, hidden_dim], cell state fir t=seq_len
        if self.rnn_cell_type == 'lstm':
            a, (h_n, c_n) = self.lstm(embed)
        else:
            a, h_n = self.gru(embed)

        # Attention
        # w: [num_directions*hidden_dim, num_directions*hidden_dim]
        u = self.tanh(torch.matmul(a, self.w) + self.b)  # [batch_size, seq_len, num_directions*hidden_dim]
        s = F.softmax(torch.matmul(u, self.c), dim=1)  # [batch_size, seq_len, 1]
                
        # Combine RNN output a and scores s
        z = torch.matmul(a.permute(0,2,1), s)  # [batch_size, num_directions*hidden_dim, 1]
        z = z.squeeze(2)  # [batch_size, num_directions*hidden_dim]
        
        z = self.dropout(z)  # [batch_size, num_directions*hidden_dim]
        z = self.linear(z)  # [batch_size, output_dim]
        if self.apply_bn == True:
            z = self.fc_bn(z)
        z = F.softmax(z, dim=1)  # [batch_size, output_dim]
        
        if self.output_attn == True:
            output = (z, s)
        else:
            output = z
            
        return output


#%%
class WordAttn(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, word_hidden_dim, word_num_layers, pad_idx, embed_trainable):
        
        super(WordAttn, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        if embed_trainable == False:
            self.embedding.weight.requires_grad = False  # Freeze embedding
        
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



###
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

###
class HAN(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, word_hidden_dim, word_num_layers, pad_idx, 
                 embed_trainable, batch_norm,
                 sent_hidden_dim, sent_num_layers, output_dim,
                 output_attn):
        
        super(HAN, self).__init__()
        
        self.word_attn = WordAttn(vocab_size = vocab_size,
                                  embedding_dim = embedding_dim,
                                  word_hidden_dim = word_hidden_dim,
                                  word_num_layers = word_num_layers,
                                  pad_idx = pad_idx,
                                  embed_trainable = embed_trainable)
        
        self.sent_attn = SentAttn(word_hidden_dim = word_hidden_dim,
                                  sent_hidden_dim = sent_hidden_dim,
                                  sent_num_layers = sent_num_layers)
        
        self.linear = nn.Linear(2*sent_hidden_dim, output_dim)
        self.fc_bn = nn.BatchNorm1d(output_dim)
        self.apply_bn = batch_norm
        self.output_attn = output_attn
    
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
        if self.apply_bn == True:
            z = self.fc_bn(z)
        z = F.softmax(z, dim=1)
        
        if self.output_attn == True:
            output = (z, self.doc_a)
        else:
            output = z
        
        return output
    

#%%
class DistilClsConv(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.distilbert = DistilBertModel(config)
        
        self.convs = nn.ModuleList([nn.Conv2d(in_channels = 1, 
                                              out_channels = 100,
                                              kernel_size = (fsize, config.dim)) for fsize in [3,4,5]
                                   ])
       
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.seq_classif_dropout)        
        self.linear = nn.Linear(100*3, config.num_labels)     
        self.softmax = nn.Softmax(dim=1)

        self.init_weights()

    def forward(
        self,
        input_ids=None, attention_mask=None,
        head_mask=None, inputs_embeds=None, labels=None,
        output_attentions=None, output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        distil_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_state = distil_output[0]  # [batch_size, seq_len, hidden_dim]
        hidden_state = hidden_state.unsqueeze(1)  # [batch_size, 1, seq_len, hidden_dim]
        
        hidden_conved = [self.relu(conv(hidden_state)) for conv in self.convs]  # hidden_conved[i]: [batch_size, n_filter, (seq_len-fsize+1), 1]
        hidden_conved = [hc.squeeze(3) for hc in hidden_conved]  # hidden_conved[i]: [batch_size, n_filter, (seq_len-fsize+1)]
        
        hc_pooled = [nn.MaxPool1d(kernel_size=hc.shape[2])(hc) for hc in hidden_conved]  # hidden_conved[i]: [batch_size, n_filter, 1]
        hc_pooled = [hc.squeeze(2) for hc in hc_pooled]  # hidden_conved[i]: [batch_size, n_filter]
        
        cat_output = torch.cat(hc_pooled, dim=1)  # [batch_size, n_filter*len(filter_sizes)]
        
        cat_output = nn.ReLU()(cat_output)  
        cat_output = self.dropout(cat_output)  
        
        logits = self.linear(cat_output)  # [batch_size, output_dim]
        probs = self.softmax(logits)   # [batch_size, output_dim]

        return probs