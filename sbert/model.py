#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 16:42:32 2020

@author: qwang
"""

import torch.nn as nn

from transformers import DistilBertModel, DistilBertPreTrainedModel
from transformers import BertModel, BertPreTrainedModel


#%%
class DistilClsLinear(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.distilbert = DistilBertModel(config)
        self.pre_linear = nn.Linear(config.dim, config.dim)
        self.linear = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)
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
        pooled_output = hidden_state[:, 0]  # [batch_size, hidden_dim]
        
        pooled_output = self.pre_linear(pooled_output)  # [batch_size, hidden_dim]
        pooled_output = nn.ReLU()(pooled_output)  
        pooled_output = self.dropout(pooled_output)  
        logits = self.linear(pooled_output)  # [batch_size, hidden_dim]
        probs = self.softmax(logits)   # [batch_size, hidden_dim]

        return probs
    
    
#%%
class BertClsLinear(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Linear(config.hidden_size, config.num_labels)
        self.softmax = nn.Softmax(dim=1)

        self.init_weights()

    def forward(
        self,
        input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
        head_mask=None, inputs_embeds=None, labels=None,
        output_attentions=None, output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]  # [batch_size, hidden_dim]

        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output)
        probs = self.softmax(logits)   # [batch_size, hidden_dim]
     
        return probs