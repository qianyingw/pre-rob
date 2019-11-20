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
    
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):
    
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(in_channels = 1,
                                              out_channels = n_filters,
                                              kernel_size = (fsize, embedding_dim)) for fsize in filter_sizes
                                   ])                
        self.fc = nn.Linear(n_filters * len(filter_sizes), output_dim)
        self.dropout = nn.Dropout(dropout)
    
    
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
        
        return out



def metrics(preds, y):
    """
    Params:
        preds: torch tensor, [batch_size, output_dim]
        y: torch tensor, [batch_size]
        
    Yields:
        A dictionary of accuracy, f1 score, recall, precision and specificity       
        
    """   
    y_preds = preds.argmax(dim=1, keepdim=False)  # [batch_size, output_dim]  --> [batch_size]
        
    ones = torch.ones_like(y_preds)
    zeros = torch.zeros_like(y_preds)
    
    pos = torch.eq(y_preds, y).sum().item()
    tp = (torch.eq(y_preds, ones) & torch.eq(y, ones)).sum().item()
    tn = (torch.eq(y_preds, zeros) & torch.eq(y, zeros)).sum().item()
    fp = (torch.eq(y_preds, ones) & torch.eq(y, zeros)).sum().item()
    fn = (torch.eq(y_preds, zeros) & torch.eq(y, ones)).sum().item()
    
    assert pos == tp + tn
    
    acc = pos / y.shape[0]  # torch.FloatTensor([y.shape[0]])
    f1 = 2*tp / (tp + tn + fp + fn) if (tp + tn + fp + fn != 0) else float('nan')
    rec = tp / (tp + fn) if (tp + fn != 0) else float('nan')
    ppv = tp / (tp + fp) if (tp + fp != 0) else float('nan')
    spc = tn / (tn + fp) if (tn + fp != 0) else float('nan')
    
    return {'accuracy': acc, 'f1': f1, 'recall': rec, 'precision': ppv, 'specificity': spc}

def confusion(preds, y):
    """
    Params:
        preds: torch tensor, [batch_size, output_dim]
        y: torch tensor, [batch_size]
        
    Yields:
        4 counts of true positive, true negative, false positive, false negative       
        
    """   
    y_preds = preds.argmax(dim=1, keepdim=False)  # [batch_size, output_dim]  --> [batch_size]
        
    ones = torch.ones_like(y_preds)
    zeros = torch.zeros_like(y_preds)

    tp = (torch.eq(y_preds, ones) & torch.eq(y, ones)).sum().item()
    tn = (torch.eq(y_preds, zeros) & torch.eq(y, zeros)).sum().item()
    fp = (torch.eq(y_preds, ones) & torch.eq(y, zeros)).sum().item()
    fn = (torch.eq(y_preds, zeros) & torch.eq(y, ones)).sum().item() 
    
    return tp, tn, fp, fn