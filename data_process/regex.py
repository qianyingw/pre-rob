#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 16:28:49 2019

@author: qwang
"""

import re


#%%
# Read regex string
def read_regex(regex_path):
    with open(regex_path, "r", encoding='utf-8') as fin:
        regex = fin.read()
    return regex


# Document annotation
def doc_annotate(regex, doc):
    match = re.search(regex, doc)
    if match:
        doc_label = 1
    else:
        doc_label = 0
    return doc_label



#%%
# Sentence tokenisation and annotation
#def sent_annotate(regex, doc):
#    sent_label = []
#    sent_list = nltk.sent_tokenize(doc)
#    for i, sent in enumerate(sent_list):
#        match = re.search(regex, sent)
#        if match:
#            sent_label.append([sent, 1])
#        else:
#            sent_label.append([sent, 0])
#    return sent_label

##%% test
#test_str = "When calculating the average latency, the cut-off time was assigned to the normal responses. The average latency was taken as ameasure for the severity of cold allodynia; shorter tail withdrawal latency was interpreted as more severe allodynia. All behavioral tests were performed in blinded fashion."
#sent_labeled = sent_annotate(regex_blinding, test_str)
#sent_labeled[0]    # [sentence, label]
#sent_labeled[0][0] # sentence
#sent_labeled[0][1] # label




