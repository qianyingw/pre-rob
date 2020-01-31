# -*- coding: utf-8 -*-
"""
Text processing and tokenization
Created on Thu Sep 26 19:17:12 2019
@author: qwang
"""

import re
import spacy


nlp = spacy.load('en')
# nlp = spacy.load('en_core_web_sm')


def preprocess_text(text):
    
    # Remove texts before the 1st occurence of introduction
    text = re.sub(r"^(.*?)\bIntroduction\b",  " ",  text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove texts after conclusions   
    text = re.sub(r"Conclusion\s{0,}\n.*|Conclusions\s{0,}\n.*",  
                  " ",  text, flags=re.DOTALL | re.IGNORECASE)
    # Remove references   
    text = re.sub(r"Reference\s{0,}\n.*|References\s{0,}\n.*|Reference list\s{0,}\n.*",  
                  " ",  text, flags=re.DOTALL | re.IGNORECASE) 
    # Remove citations 
    text = re.sub(r"\s+[\[][^a-zA-Z]+[\]]", "", text)
    # Remove emtpy lines
    text = re.sub(r"^(?:[\t ]*(?:\r?\n|\r))+", " ", text, flags=re.MULTILINE)
    # Remove lines with digits/(digits,punctuations,line character) only
    text = re.sub(r"^\W{0,}\d{1,}\W{0,}$", "", text)
    # Remove non-ascii characters
    text = text.encode("ascii", errors="ignore").decode()     
    # Strip whitespaces 
    text = re.sub(r'\s+', ' ', text)  
 
    return text.lower()


# Tokenization
#def tokenize_text(text):
#    tokens = [token.text for token in nlp(text)]
#    return tokens
    
from nltk.tokenize import sent_tokenize, word_tokenize
def word_tokenizer(text):
    tokens = word_tokenize(text)
    return tokens

def sent_tokenizer(text):
    tokens = sent_tokenize(text)
    return tokens
