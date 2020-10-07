#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 15:41:18 2020

@author: qwang
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertPreTrainedModel, BertConfig, BertModel


#%%
class BertClsPLinear(BertPreTrainedModel):

    def __init__(self, config: BertConfig):
        super().__init__(config)
        
        self.config = config
        self.bert = BertModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        if config.pool_method_chunks == 'mean_max':
            self.fc = nn.Linear(config.hidden_size*2, config.num_labels)
        else:
            self.fc = nn.Linear(config.hidden_size, config.num_labels)
            
        self.fc_bn = nn.BatchNorm1d(config.num_labels)
        self.init_weights()
        
        # Default: freeze bert
        for name, param in self.bert.named_parameters():
            param.requires_grad = False  

        # Unfreeze layers                   
        if config.unfreeze == "pooler":
            for name, param in self.bert.named_parameters():
                if "pooler" in name:
                    param.requires_grad = True 
                    
        if config.unfreeze == "enc-1":
            n_layer = sum([1 for name, _ in self.bert.named_parameters() if "encoder.layer" in name])
            last_layer = "encoder.layer." + str(int(n_layer/16-1))  # each enc layer has 16 pars
            for name, param in self.bert.named_parameters():               
                if last_layer in name:
                    param.requires_grad = True
        
        if config.unfreeze == "enc-1_pooler":
            n_layer = sum([1 for name, _ in self.bert.named_parameters() if "encoder.layer" in name])
            last_layer = "encoder.layer." + str(int(n_layer/16-1))  # each enc layer has 16 pars
            for name, param in self.bert.named_parameters():               
                if last_layer in name or "pooler" in name:
                    param.requires_grad = True
        
    def forward(self, doc):
        """
        Input:
            doc: [batch_size, n_chunks, 3, max_chunk_len]     
                 n_chunks is the number of chunks within the batch (same for each doc after PadDoc)
        Returns:
            out: [batch_size, output_dim]       
            
        """    
        batch_size = doc.shape[0]          
        
        hidden_pooled_layers = []
        
        for k in range(batch_size):
            # Each doc is considered as a special 'batch' and each chunk is an element of the special 'bert_batch'
            # n_chunks is the temporary 'bert_batch_size', max_chunk_len corresponds to 'seq_len'
            bert_output_k = self.bert(input_ids = doc[k,:,0],  # [n_chunks, max_chunk_len]
                                      attention_mask = doc[k,:,1], 
                                      token_type_ids = doc[k,:,2])
            # pooled_k = bert_output_k[1].unsqueeze(0) 
            hidden_states_k = bert_output_k[2]  # each element in the tuple: [n_chunks, max_chunk_len, hidden_size]
             
            # Average pooling over last [pool_layers] layers        
            hidden_list_k = list(hidden_states_k[self.config.pool_layers:])
            hidden_stack_k = torch.stack(hidden_list_k)  # [n_pooled_layers, n_chunks, max_chunk_len, hidden_size]                    
            hidden_pooled_layers_k = torch.mean(hidden_stack_k, dim=0)  # [n_chunks, max_chunk_len, hidden_size]
            hidden_pooled_layers.append(hidden_pooled_layers_k)
    
        
        hidden_pooled_layers = torch.stack(hidden_pooled_layers)  # [batch_size, n_chunks, max_chunk_len, hidden_size]
        # Pooling within each chunk (over 512 word tokens of individual chunk)
        if self.config.pool_method == 'mean':
            hidden_pooled = torch.mean(hidden_pooled_layers, dim=2)  # [batch_size, n_chunks, hidden_size]
        elif self.config.pool_method == 'max':
            hidden_pooled = torch.max(hidden_pooled_layers, dim=2).values  # [batch_size, n_chunks, hidden_size]
        elif self.config.pool_method == 'mean_max':
            hidden_pooled_mean = torch.mean(hidden_pooled_layers, dim=2)               # [batch_size, n_chunks, hidden_size]
            hidden_pooled_max = torch.max(hidden_pooled_layers, dim=2).values          # [batch_size, n_chunks, hidden_size]
            hidden_pooled = torch.cat((hidden_pooled_mean, hidden_pooled_max), dim=1)  # [batch_size, n_chunks*2, hidden_size]
        elif self.config.pool_method == 'cls':
            hidden_pooled = hidden_pooled_layers[:,:,0,:]  # [batch_size, n_chunks, hidden_size]
        else: # pool_method is None
            hidden_pooled = hidden_pooled_layers.view(batch_size, -1, self.config.hidden_size) # [batch_size, n_chunks*max_chunk_len, hidden_size]
            
       
        dp = self.dropout(hidden_pooled)   # [batch_size, ?, hidden_size]
                                           # ? can be n_chunks, n_chunks*2 or n_chunks*max_chunk_len)            
        # Pooling over chunks
        if self.config.pool_method_chunks == 'mean':
            dp_pooled = torch.mean(dp, dim=1)  # [batch_size, hidden_size]
        if self.config.pool_method_chunks == 'max':
            dp_pooled = torch.max(dp, dim=1).values  # [batch_size, hidden_size]
        if self.config.pool_method_chunks == 'mean_max':
            dp_pooled_mean = torch.mean(dp, dim=1)  # [batch_size, hidden_size]
            dp_pooled_max = torch.max(dp, dim=1).values  # [batch_size, hidden_size]
            dp_pooled = torch.cat((dp_pooled_mean, dp_pooled_max), dim=1)  # [batch_size, 2*hidden_size]
        if self.config.pool_method_chunks == 'cls':
            dp_pooled = dp[:,0,:]  # [batch_size, hidden_size]
      
        out = self.fc(dp_pooled)  # [batch_size, num_labels]   
        out = self.fc_bn(out)
        out = F.softmax(out, dim=1)  # [batch_size, num_labels]      
             
        return out
    
#%%
class BertClsPLSTM(BertPreTrainedModel):

    def __init__(self, config: BertConfig):
        super().__init__(config)
        
        self.config = config
        self.bert = BertModel(config)

        # self.seq_summary = SequenceSummary(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        
        self.lstm = nn.LSTM(input_size = config.hidden_size, hidden_size = config.hidden_size,
                            num_layers = 1, dropout = 0, 
                            batch_first = True, bidirectional = False)
             
        self.fc = nn.Linear(config.hidden_size, config.num_labels)
        self.fc_bn = nn.BatchNorm1d(config.num_labels)
        self.init_weights()
        
        # Default: freeze bert
        for name, param in self.bert.named_parameters():
            param.requires_grad = False  

        # Unfreeze layers                   
        if config.unfreeze == "pooler":
            for name, param in self.bert.named_parameters():
                if "pooler" in name:
                    param.requires_grad = True 
                    
        if config.unfreeze == "enc-1":
            n_layer = sum([1 for name, _ in self.bert.named_parameters() if "encoder.layer" in name])
            last_layer = "encoder.layer." + str(int(n_layer/16-1))  # each enc layer has 16 pars
            for name, param in self.bert.named_parameters():               
                if last_layer in name:
                    param.requires_grad = True
        
        if config.unfreeze == "enc-1_pooler":
            n_layer = sum([1 for name, _ in self.bert.named_parameters() if "encoder.layer" in name])
            last_layer = "encoder.layer." + str(int(n_layer/16-1))  # each enc layer has 16 pars
            for name, param in self.bert.named_parameters():               
                if last_layer in name or "pooler" in name:
                    param.requires_grad = True
        
    def forward(self, doc):
        """
        Input:
            doc: [batch_size, n_chunks, 3, max_chunk_len]     
                 n_chunks is the number of chunks within the batch (same for each doc after PadDoc)
        Returns:
            out: [batch_size, output_dim]       
            
        """    
        batch_size = doc.shape[0]          
        
        hidden_pooled_layers = []
        
        for k in range(batch_size):
            # Each doc is considered as a special 'batch' and each chunk is an element of the special 'bert_batch'
            # n_chunks is the temporary 'bert_batch_size', max_chunk_len corresponds to 'seq_len'
            bert_output_k = self.bert(input_ids = doc[k,:,0],  # [n_chunks, max_chunk_len]
                                      attention_mask = doc[k,:,1], 
                                      token_type_ids = doc[k,:,2])
            # pooled_k = bert_output_k[1].unsqueeze(0) 
            hidden_states_k = bert_output_k[2]  # each element in the tuple: [n_chunks, max_chunk_len, hidden_size]
             
            # Average pooling over last [pool_layers] layers        
            hidden_list_k = list(hidden_states_k[self.config.pool_layers:])
            hidden_stack_k = torch.stack(hidden_list_k)  # [n_pooled_layers, n_chunks, max_chunk_len, hidden_size]                    
            hidden_pooled_layers_k = torch.mean(hidden_stack_k, dim=0)  # [n_chunks, max_chunk_len, hidden_size]
            hidden_pooled_layers.append(hidden_pooled_layers_k)
    
        
        hidden_pooled_layers = torch.stack(hidden_pooled_layers)  # [batch_size, n_chunks, max_chunk_len, hidden_size]
        # Pooling within each chunk (over 512 word tokens of individual chunk)
        if self.config.pool_method == 'mean':
            hidden_pooled = torch.mean(hidden_pooled_layers, dim=2)  # [batch_size, n_chunks, hidden_size]
        elif self.config.pool_method == 'max':
            hidden_pooled = torch.max(hidden_pooled_layers, dim=2).values  # [batch_size, n_chunks, hidden_size]
        elif self.config.pool_method == 'mean_max':
            hidden_pooled_mean = torch.mean(hidden_pooled_layers, dim=2)               # [batch_size, n_chunks, hidden_size]
            hidden_pooled_max = torch.max(hidden_pooled_layers, dim=2).values          # [batch_size, n_chunks, hidden_size]
            hidden_pooled = torch.cat((hidden_pooled_mean, hidden_pooled_max), dim=1)  # [batch_size, n_chunks*2, hidden_size]
        elif self.config.pool_method == 'cls':
            hidden_pooled = hidden_pooled_layers[:,:,0,:]  # [batch_size, n_chunks, hidden_size]
        else: # pool_method is None
            hidden_pooled = hidden_pooled_layers.view(batch_size, -1, self.config.hidden_size) # [batch_size, n_chunks*max_chunk_len, hidden_size]
            
    
       
        dp = self.dropout(hidden_pooled)   # [batch_size, ?, hidden_size]
        # ? can be n_chunks, n_chunks*2 or n_chunks*max_chunk_len)      
        # output: [batch_size, ?, n_directions*hidden_size], output features from last layer for each t
        # h_n: [n_layers*n_directions, batch_size, hidden_size], hidden state for t=seq_len
        # c_n: [n_layers*n_directions, batch_size, hidden_size], cell state fir t=seq_len
        lstm_output, (h_n, c_n) = self.lstm(dp)
           
        pooled_lstm = lstm_output[:,0,:]  # [batch_size, hidden_size]

    
        out = self.fc(pooled_lstm)  # [batch_size, num_labels]   
        out = self.fc_bn(out)
        out = F.softmax(out, dim=1)  # [batch_size, num_labels]      
             
        return out



#%%
class BertClsPConv(BertPreTrainedModel):

    def __init__(self, config: BertConfig):
        super().__init__(config)
        
        self.config = config
        self.bert = BertModel(config)

        # self.seq_summary = SequenceSummary(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.convs = nn.ModuleList([nn.Conv2d(in_channels = 1,
                                              out_channels = config.num_filters,
                                              kernel_size = (fsize, config.hidden_size)) for fsize in config.filter_sizes
                                   ])       

        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_labels)
        self.fc_bn = nn.BatchNorm1d(config.num_labels)
        self.init_weights()
        
        # Default: freeze bert
        for name, param in self.bert.named_parameters():
            param.requires_grad = False  

        # Unfreeze layers                   
        if config.unfreeze == "pooler":
            for name, param in self.bert.named_parameters():
                if "pooler" in name:
                    param.requires_grad = True 
                    
        if config.unfreeze == "enc-1":
            n_layer = sum([1 for name, _ in self.bert.named_parameters() if "encoder.layer" in name])
            last_layer = "encoder.layer." + str(int(n_layer/16-1))  # each enc layer has 16 pars
            for name, param in self.bert.named_parameters():               
                if last_layer in name:
                    param.requires_grad = True
        
        if config.unfreeze == "enc-1_pooler":
            n_layer = sum([1 for name, _ in self.bert.named_parameters() if "encoder.layer" in name])
            last_layer = "encoder.layer." + str(int(n_layer/16-1))  # each enc layer has 16 pars
            for name, param in self.bert.named_parameters():               
                if last_layer in name or "pooler" in name:
                    param.requires_grad = True
        
    def forward(self, doc):
        """
        Input:
            doc: [batch_size, n_chunks, 3, max_chunk_len]     
                 n_chunks is the number of chunks within the batch (same for each doc after PadDoc)
        Returns:
            out: [batch_size, output_dim]       
            
        """    
        batch_size = doc.shape[0]          
        
        hidden_pooled_layers = []
        
        for k in range(batch_size):
            # Each doc is considered as a special 'batch' and each chunk is an element of the special 'bert_batch'
            # n_chunks is the temporary 'bert_batch_size', max_chunk_len corresponds to 'seq_len'
            bert_output_k = self.bert(input_ids = doc[k,:,0],  # [n_chunks, max_chunk_len]
                                      attention_mask = doc[k,:,1], 
                                      token_type_ids = doc[k,:,2])
            # pooled_k = bert_output_k[1].unsqueeze(0) 
            hidden_states_k = bert_output_k[2]  # each element in the tuple: [n_chunks, max_chunk_len, hidden_size]
             
            # Average pooling over last [pool_layers] layers        
            hidden_list_k = list(hidden_states_k[self.config.pool_layers:])
            hidden_stack_k = torch.stack(hidden_list_k)  # [n_pooled_layers, n_chunks, max_chunk_len, hidden_size]                    
            hidden_pooled_layers_k = torch.mean(hidden_stack_k, dim=0)  # [n_chunks, max_chunk_len, hidden_size]
            hidden_pooled_layers.append(hidden_pooled_layers_k)
    
        
        hidden_pooled_layers = torch.stack(hidden_pooled_layers)  # [batch_size, n_chunks, max_chunk_len, hidden_size]
        # Pooling within each chunk (over 512 word tokens of individual chunk)
        if self.config.pool_method == 'mean':
            hidden_pooled = torch.mean(hidden_pooled_layers, dim=2)  # [batch_size, n_chunks, hidden_size]
        elif self.config.pool_method == 'max':
            hidden_pooled = torch.max(hidden_pooled_layers, dim=2).values  # [batch_size, n_chunks, hidden_size]
        elif self.config.pool_method == 'mean_max':
            hidden_pooled_mean = torch.mean(hidden_pooled_layers, dim=2)               # [batch_size, n_chunks, hidden_size]
            hidden_pooled_max = torch.max(hidden_pooled_layers, dim=2).values          # [batch_size, n_chunks, hidden_size]
            hidden_pooled = torch.cat((hidden_pooled_mean, hidden_pooled_max), dim=1)  # [batch_size, n_chunks*2, hidden_size]
        elif self.config.pool_method == 'cls':
            hidden_pooled = hidden_pooled_layers[:,:,0,:]  # [batch_size, n_chunks, hidden_size]
        else: # pool_method is None
            hidden_pooled = hidden_pooled_layers.view(batch_size, -1, self.config.hidden_size)  # [batch_size, n_chunks*max_chunk_len, hidden_size]
            
        hidden_pooled = hidden_pooled.unsqueeze(1)  # [batch_size, 1, ?, hidden_size]
        hidden_conved = [F.relu(conv(hidden_pooled)) for conv in self.convs]  # hidden_conved[i]: [batch_size, n_filters, (?-fsize+1), 1]
        hidden_conved = [conv.squeeze(3) for conv in hidden_conved]  # hidden_conved[i]: [batch_size, n_filters, (?-fsize+1)]
        hc_pooled = [F.max_pool1d(conv, conv.shape[2]) for conv in hidden_conved]  # hc_pooled[i]: [batch_size, n_filters, 1]
        hc_pooled = [pool.squeeze(2) for pool in hc_pooled]  # hc_pooled[i]: [batch_size, n_filters]
        
        cat = torch.cat(hc_pooled, dim=1)  # [batch_size, n_filters * len(filter_sizes)]
        dp = self.dropout(cat)
        out = self.fc(dp)  # # [batch_size, num_labels]
        out = self.fc_bn(out)
        out = F.softmax(out, dim=1)  # [batch_size, num_labels]  

             
        return out