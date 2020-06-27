#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 20:33:16 2020
@author: qwang
Ref: https://github.com/Lynten/stanford-corenlp
"""

from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP(r'/home/qwang/stanford-corenlp-full-2018-10-05')

# Simple usage
sentence = 'Guangdong University of Foreign Studies is located in Guangzhou.'
w_tokens = nlp.word_tokenize(sentence)

# General Stanford CoreNLP API
text = 'Guangdong University of Foreign Studies is located in Guangzhou. ' \
       'GDUFS is active in a full range of international cooperation and exchanges in education. '

props={'annotators':'tokenize,ssplit', 'pipelineLanguage':'en', 'outputFormat':'json'}
doc = nlp.annotate(text, properties=props)


temp['sentences']
temp.sentences[0]
nlp.close() # Do not forget to close! The backend server will consume a lot memery.


#%%
import stanfordnlp
# Downloads the English models for the neural pipeline
#stanfordnlp.download('en', force=True)   

nlp = stanfordnlp.Pipeline(lang='en', processors='tokenize', use_gpu=False)
# Processing English text
doc = nlp("Barack Obama was born in Hawaii.  He was elected president in 2008. Yoongi is hungry!")


sent_tokens = []
word_tokens = []
for i, sent in enumerate(doc.sentences):
    s = []
    for word in sent.words:
        s.append(word.text)
        word_tokens.append(word.text)
    sent_tokens.append(s)

print(sent_tokens)
print(word_tokens)

    

#%%
import re
from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')

properties={
  'annotators': 'ssplit',
  'outputFormat': 'json'
  }

def sentence_split(text, properties={'annotators': 'ssplit', 'outputFormat': 'json'}):
    """Split sentence using Stanford NLP"""
    annotated = nlp.annotate(text, properties)
    sentence_split = list()
    for sentence in annotated['sentences']:
        s = [t['word'] for t in sentence['tokens']]
        sentence_split.append(s)
    return sentence_split

text = 'Hello all. My name is Titipat, the best LoL player.'
sentence_split(text, properties)
