#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 14:56:39 2020

@author: qwang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertPreTrainedModel, BertConfig, BertModel
from transformers import AlbertPreTrainedModel, AlbertConfig, AlbertModel

# from hgf.modeling_utils import SequenceSummary

#%%
class BertPoolLSTM(BertPreTrainedModel):

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
        output, (h_n, c_n) = self.lstm(dp)
           
        h_n = h_n.squeeze(0)  # [batch_size, hidden_size]

    
        out = self.fc(h_n)  # [batch_size, num_labels]   
        out = self.fc_bn(out)
        out = F.softmax(out, dim=1)  # [batch_size, num_labels]      
             
        return out



#%%
class BertPoolConv(BertPreTrainedModel):

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

#%%
class BertLSTM(BertPreTrainedModel):

    def __init__(self, config: BertConfig):
        super().__init__(config)
        
        self.bert = BertModel(config)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
                
        self.lstm = nn.LSTM(input_size = config.hidden_size, hidden_size = config.hidden_size,
                            num_layers = 1, dropout = 0, 
                            batch_first = True, bidirectional = False)
        
        self.fc = nn.Linear(config.hidden_size*3, config.num_labels)
        self.fc_bn = nn.BatchNorm1d(config.num_labels)
        self.tanh = nn.Tanh()
        self.init_weights()
        
        # Default: freeze bert
        for name, param in self.bert.named_parameters():
            param.requires_grad = False  

        # Unfreeze layers
        if config.unfreeze == "embed":
            for name, param in self.bert.named_parameters():
                if "embeddings" in name:
                    param.requires_grad = True 
                    
        if config.unfreeze == "embed_enc0":
            for name, param in self.bert.named_parameters():
                if "embeddings" in name or "encoder.layer.0" in name:
                    param.requires_grad = True
                    
        if config.unfreeze == "embed_enc0_pooler":
            for name, param in self.bert.named_parameters():
                if "embeddings" in name or "encoder.layer.0" in name or "pooler" in name:
                    param.requires_grad = True 
                    
        if config.unfreeze == "enc0":
            for name, param in self.bert.named_parameters():
                if "encoder.layer.0" in name:
                    param.requires_grad = True 
                    
        if config.unfreeze == "enc0_pooler":
            for name, param in self.bert.named_parameters():
                if "encoder.layer.0" in name or "pooler" in name:
                    param.requires_grad = True
        
        if config.unfreeze == "embed_pooler":
            for name, param in self.bert.named_parameters():
                if "embed" in name or "pooler" in name:
                    param.requires_grad = True 
                    
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
            doc: [batch_size, num_chunks, 3, max_chunk_len]            
        Returns:
            out: [batch_size, output_dim]       
            
        """    
        batch_size = doc.shape[0]        
        
        pooled = self.bert(input_ids = doc[0,:,0], 
                           attention_mask = doc[0,:,1], 
                           token_type_ids = doc[0,:,2])[1].unsqueeze(0) 
        
        for i in range(batch_size-1):
            # Output of BertModel: (last_hidden_state, pooler_output, hidden_states, attentions)
            # Last layer hidden-state of the first token of the sequence (classification token)
            pool_i = self.bert(input_ids = doc[i+1,:,0], 
                               attention_mask = doc[i+1,:,1], 
                               token_type_ids = doc[i+1,:,2])[1]
            pooled = torch.cat((pooled, pool_i.unsqueeze(0)), dim=0)
            
        
        dp = self.dropout(pooled)  # [batch_size, num_chunks, hidden_size]
        # output: [batch_size, num_chunks, n_directions*hidden_size], output features from last layer for each t
        # h_n: [n_layers*n_directions, batch_size, hidden_size], hidden state for t=seq_len
        # c_n: [n_layers*n_directions, batch_size, hidden_size], cell state fir t=seq_len
        output, (h_n, c_n) = self.lstm(dp)
        
        
        # Concat pooling
        # h_n = output[:,-1,].squeeze(1)  # [batch_size, hidden_size]
        h_n = h_n.squeeze(0)  # [batch_size, hidden_size]
        h_max = torch.max(output, dim=1).values  # [batch_size, hidden_size]
        h_mean = torch.mean(output, dim=1)  # [batch_size, hidden_size]
        out = torch.cat((h_n, h_max, h_mean), dim=1)  # [batch_size, hidden_size*3]
        
        out = self.fc(out)  # [batch_size, num_labels]   
        out = self.fc_bn(out)
        out = F.softmax(out, dim=1)  # [batch_size, num_labels]
        # out = self.tanh(out)   # [batch_size, num_labels]
        
        return out


#%%
class AlbertLinear(AlbertPreTrainedModel):

    def __init__(self, config: AlbertConfig):
        super().__init__(config)
        
        self.albert = AlbertModel(config)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.fc = nn.Linear(config.hidden_size, config.num_labels)
        self.fc_bn = nn.BatchNorm1d(config.num_labels)
        # self.fc = nn.Linear(config.hidden_size * config.n_chunks, config.num_labels)
        self.init_weights()
        
        # Default: freeze albert
        for name, param in self.albert.named_parameters():
            param.requires_grad = False  

        # Unfreeze layers
        if config.unfreeze == "embed":
            for name, param in self.albert.named_parameters():
                if "embeddings" in name:
                    param.requires_grad = True 
                    
        if config.unfreeze == "embed_enc0":
            for name, param in self.albert.named_parameters():
                if "embeddings" in name or "encoder" in name:
                    param.requires_grad = True
                    
        if config.unfreeze == "embed_enc0_pooler":
            for name, param in self.albert.named_parameters():
                    param.requires_grad = True 
                    
        if config.unfreeze == "enc0":
            for name, param in self.albert.named_parameters():
                if "encoder" in name:
                    param.requires_grad = True 
                    
        if config.unfreeze == "enc0_pooler":
            for name, param in self.albert.named_parameters():
                if "encoder" in name or "pooler" in name:
                    param.requires_grad = True
        
        if config.unfreeze == "embed_pooler":
            for name, param in self.albert.named_parameters():
                if "embed" in name or "pooler" in name:
                    param.requires_grad = True 
                    
        if config.unfreeze == "pooler":
            for name, param in self.albert.named_parameters():
                if "pooler" in name:
                    param.requires_grad = True
        
     
    def forward(self, doc):
        """
        Input:
            doc: [batch_size, num_chunks, 3, max_chunk_len]            
        Returns:
            out: [batch_size, output_dim]       
            
        """    
        batch_size = doc.shape[0]        
        
        pooled = self.albert(input_ids = doc[0,:,0], 
                             attention_mask = doc[0,:,1], 
                             token_type_ids = doc[0,:,2])[1].unsqueeze(0) 
        for i in range(batch_size-1):
            pool_i = self.albert(input_ids = doc[i+1,:,0], 
                                 attention_mask = doc[i+1,:,1], 
                                 token_type_ids = doc[i+1,:,2])[1]
            pooled = torch.cat((pooled, pool_i.unsqueeze(0)), dim=0)
 
                
        dp = self.dropout(pooled)  # [batch_size, num_chunks, hidden_size]  
        # concat = dp.view(batch_size, -1)  # [batch_size, num_chunks*hidden_size]
        if self.albert.config.linear_max == True:
            dp = torch.max(dp, dim=1).values  # [batch_size, hidden_size]
        else:
            dp = torch.mean(dp, dim=1)  # [batch_size, hidden_size]
        # dp = dp.sum(dim=1) # [batch_size, hidden_size]

        out = self.fc(dp)  # [batch_size, num_labels]   
        out = self.fc_bn(out)
        out = F.softmax(out, dim=1)  # [batch_size, num_labels]
             
        return out

#%%
class AlbertLSTM(AlbertPreTrainedModel):

    def __init__(self, config: AlbertConfig):
        super().__init__(config)
        
        self.albert = AlbertModel(config)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
                
        self.lstm = nn.LSTM(input_size = config.hidden_size, hidden_size = config.hidden_size,
                            num_layers = 1, dropout = 0, 
                            batch_first = True, bidirectional = False)
        
        self.fc = nn.Linear(config.hidden_size, config.num_labels)
        self.fc_bn = nn.BatchNorm1d(config.num_labels)
        self.tanh = nn.Tanh()
        self.init_weights()  
        
        # Default: freeze albert
        for name, param in self.albert.named_parameters():
            param.requires_grad = False  

        # Unfreeze layers
        if config.unfreeze == "embed":
            for name, param in self.albert.named_parameters():
                if "embeddings" in name:
                    param.requires_grad = True 
                    
        if config.unfreeze == "embed_enc0":
            for name, param in self.albert.named_parameters():
                if "embeddings" in name or "encoder" in name:
                    param.requires_grad = True
                    
        if config.unfreeze == "embed_enc0_pooler":
            for name, param in self.albert.named_parameters():
                    param.requires_grad = True 
                    
        if config.unfreeze == "enc0":
            for name, param in self.albert.named_parameters():
                if "encoder" in name:
                    param.requires_grad = True 
                    
        if config.unfreeze == "enc0_pooler":
            for name, param in self.albert.named_parameters():
                if "encoder" in name or "pooler" in name:
                    param.requires_grad = True
        
        if config.unfreeze == "embed_pooler":
            for name, param in self.albert.named_parameters():
                if "embed" in name or "pooler" in name:
                    param.requires_grad = True 
                    
        if config.unfreeze == "pooler":
            for name, param in self.albert.named_parameters():
                if "pooler" in name:
                    param.requires_grad = True
                                   
    
    def forward(self, doc):
        """
        Input:
            doc: [batch_size, num_chunks, 3, max_chunk_len]            
        Returns:
            out: [batch_size, output_dim]       
            
        """    
        batch_size = doc.shape[0]        
        
        pooled = self.albert(input_ids = doc[0,:,0], 
                             attention_mask = doc[0,:,1], 
                             token_type_ids = doc[0,:,2])[1].unsqueeze(0) 
        
        for i in range(batch_size-1):
            # Output of BertModel: (last_hidden_state, pooler_output, hidden_states, attentions)
            # Last layer hidden-state of the first token of the sequence (classification token)
            pool_i = self.albert(input_ids = doc[i+1,:,0], 
                                 attention_mask = doc[i+1,:,1], 
                                 token_type_ids = doc[i+1,:,2])[1]
            pooled = torch.cat((pooled, pool_i.unsqueeze(0)), dim=0)
            
        
        dp = self.dropout(pooled)  # [batch_size, num_chunks, bert_hidden_size]
        # output: [batch_size, num_chunks, n_directions*hidden_size], output features from last layer for each t
        # h_n: [n_layers*n_directions, batch_size, hidden_size], hidden state for t=seq_len
        # c_n: [n_layers*n_directions, batch_size, hidden_size], cell state fir t=seq_len
        output, (h_n, c_n) = self.lstm(dp)
        
        
        # h_n = output[:,-1,].squeeze(1)  # [batch_size, hidden_size]
        h_n = h_n.squeeze(0)  # [batch_size, hidden_size]
        
        out = self.fc(h_n)  # [batch_size, num_labels]  
        out = self.fc_bn(out)  
        out = F.softmax(out, dim=1)  # [batch_size, num_labels]
        # out = self.tanh(out)   # [batch_size, num_labels]
        
        return out