#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 17:12:41 2019

@author: qwang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Define the neural network module and metrics

Created on Fri Oct  4 11:07:31 2019
@author: qwang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


#net = ConvNet(10000, 200, 100, [3,4,5], 2, 0.5, 1, True)
#for p in net.parameters():
#    print(p.size())
    

#%%
class RecurNet(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, rnn_hidden_dim, rnn_num_layers, output_dim, bidirection, rnn_cell_type, dropout, pad_idx, embed_trainable, batch_norm):
       
        super(RecurNet, self).__init__()
        
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
        
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(rnn_hidden_dim*num_directions, output_dim)
        self.rnn_cell_type = rnn_cell_type
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
        
        # out: [batch_size, seq_len, num_directions*hidden_dim], output features from last layer for each t
        # h_n: [batch_size, num_layers*num_directions, hidden_dim], hidden state for t=seq_len
        # c_n: [batch_size, num_layers*num*directions, hidden_dim], cell state fir t=seq_len
        if self.rnn_cell_type == 'lstm':
            out, (h_n, c_n) = self.lstm(embed)
        else:
            out, h_n = self.gru(embed)

        # Obtain the last hidden state from last layer
        batch_len = out.size()[0]
        out_list = []
        for i in range(batch_len):
            # Append hidden state of last word from each batch
            # out[i][-1]: [num_directions*hidden_dim]
            out_list.append(out[i][-1])  
        out_last = torch.stack(out_list)  # [batch_size, num_directions*hidden_dim]
        # Next: out_max, out_attn
        
        z = self.dropout(out_last)  # [batch_size, num_directions*hidden_dim]
        z = self.linear(z)  # [batch_size, output_dim]
        if self.apply_bn == True:
            z = self.fc_bn(z)
        z = F.softmax(z, dim=1)  # [batch_size, output_dim]
        
        return z



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


#net = AttnNet(10000, 200, 30, 1, 2, True, 'lstm', 0.5, 1, True)
#for p in net.parameters():
#    print(p.size())