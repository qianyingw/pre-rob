# -*- coding: utf-8 -*-
"""
Text processing and tokenization
Created on Thu Sep 26 19:17:12 2019
@author: qwang
"""

import re
import string

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
    # Remove links
    text = re.sub(r"https?:/\/\S+", " ", text)
    # Remove emtpy lines
    text = re.sub(r"^(?:[\t ]*(?:\r?\n|\r))+", " ", text, flags=re.MULTILINE)
    # Remove lines with digits/(digits,punctuations,line character) only
    text = re.sub(r"^\W{0,}\d{1,}\W{0,}$", "", text)
    # Remove numbers
    text = re.sub(r'\d+', "", text)
    # Replace hyphens to spaces
    text = re.sub(r"-", " ", text)
    # Remove non-ascii characters
    text = text.encode("ascii", errors="ignore").decode()     
    # Strip whitespaces 
    text = re.sub(r'\s+', " ", text)
    # Remove the whitespace at start and end of line
    text = re.sub(r'^[\s]', "", text)
    text = re.sub(r'[\s]$', "", text)
 
    return text.lower()



#%% Tokenization (stanford corenlp)
import stanfordnlp
# Downloads the English models for the neural pipeline
# stanfordnlp.download('en', force=True)   
nlp = stanfordnlp.Pipeline(lang='en', processors='tokenize', use_gpu=False)

def text_tokenizer(text):    
    sent_tokens = []
    word_tokens = []
    text = nlp(text)
    for i, sent in enumerate(text.sentences):
        one_sent = [word.text for word in sent.words if word.text not in string.punctuation]
        if len(one_sent) > 2:
            sent_tokens.append(one_sent)
    word_tokens = [w for s in sent_tokens for w in s]
    return sent_tokens, word_tokens



# Example
#text = "i don't like mustard. yk is hungry. he wants a banana. she's 5-social. i. not. he - not really.    9. "
#text = preprocess_text(text)
#text = nlp(text)
#sent_tokens = []
#word_tokens = []
#for i, sent in enumerate(text.sentences):
#    one_sent = [word.text for word in sent.words if word.text not in string.punctuation]
#    sent_tokens.append(one_sent)
#word_tokens = [w for s in sent_tokens for w in s]    
#    
#print(sent_tokens)
#print(word_tokens)
#len(word_tokens)


#%% Tokenization (spacy)
#    import spacy
#    nlp = spacy.load('en')
#    def tokenize_text(text):
#        tokens = [token.text for token in nlp(text)]
#        return tokens


#%% Tokenization (nltk)  
#    from nltk.tokenize import sent_tokenize, word_tokenize
#    def word_tokenizer(text):
#        tokens = word_tokenize(text)
#        return tokens
#    
#    def sent_tokenizer(text):
#        tokens = sent_tokenize(text)
#        return tokens
