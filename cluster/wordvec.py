#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 12:07:26 2020

@author: qwang
"""

from gensim.models.keyedvectors import KeyedVectors

model = KeyedVectors.load_word2vec_format('/media/mynewdrive/rob/wordvec/wikipedia-pubmed-and-PMC-w2v.bin', binary=True)
model.save_word2vec_format('/home/qwang/rob/wikipedia-pubmed-and-PMC-w2v.txt', binary=False)

