# -*- coding: utf-8 -*-
"""
Text processing and tokenization
Created on Thu Sep 26 19:17:12 2019
@author: qwang
"""

import re
import string

p_ref = re.compile(r"(.*Reference\s{0,}\n)|(.*References\s{0,}\n)|(.*Reference list\s{0,}\n)|(.*REFERENCE\s{0,}\n)|(.*REFERENCES\s{0,}\n)|(.*REFERENCE LIST\s{0,}\n)", 
                   flags=re.DOTALL)

def preprocess_text(text):
    
    # Remove texts before the first occurence of 'Introduction' or 'INTRODUCTION'
    text = re.sub(r".*?(Introduction|INTRODUCTION)\s{0,}\n{1,}", " ", text, count=1, flags=re.DOTALL)
    
    # Remove texts after conclusions (may exist in abstract..)   
    # text = re.sub(r"Conclusion\s{0,}\n.*|Conclusions\s{0,}\n.*", " ",  text, flags=re.DOTALL | re.IGNORECASE)

    # Remove reference after the last occurence 
    s = re.search(p_ref, text)
    if s: text = s[0]
    # text = re.sub(r"Reference\s{0,}\n.*|References\s{0,}\n.*|Reference list\s{0,}\n.*|REFERENCE\s{0,}\n.*|REFERENCES\s{0,}\n.*|REFERENCE LIST\s{0,}\n.*", " ", text, flags=re.DOTALL)  
    # text = re.sub(r"^(?!preference|preferences)Reference\s{0,}\n.*|References\s{0,}\n.*|Reference list\s{0,}\n.*", " ", text, flags=re.DOTALL | re.IGNORECASE)    
    # text = re.sub(r"Reference\s{0,}\n.*|References\s{0,}\n.*|Reference list\s{0,}\n.*", " ", text, flags=re.DOTALL | re.IGNORECASE) 
    
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


#%% Tokenization with sentence embedding
## stanford tokenizer
def sent_encoder(embed_func, text):    
    sent_list = []
    text = nlp(text)
    for i, sent in enumerate(text.sentences):
        one_sent_list = [word.text for word in sent.words]
        if len(one_sent_list) > 2:
            one_sent = ' '.join(one_sent_list)
            sent_list.append(one_sent)
    doc_mat = embed_func(sent_list).numpy().astype('float_')                 
    doc_mat = doc_mat.tolist()
    return doc_mat

# nltk
#from nltk.tokenize import word_tokenize, sent_tokenize
#def sent_encoder(embed_func, text):    
#    sent_list = sent_tokenize(text)
#    sent_list = [s for s in sent_list if len(word_tokenize(s)) > 2]            
#    doc_mat = embed_func(sent_list).numpy()
#    return doc_mat

### Example                   
#text = "i don't like mustard, but he likes it. yk is hungry. he wants a banana. she's 5-social. i. not. he - not really.    9. "
#text = preprocess_text(text)
#text = nlp(text)
#temp = sent_encoder(embed_func, text)
#sent_list = []
#for i, sent in enumerate(text.sentences):
#    one_sent_list = [word.text for word in sent.words if word.text]
#    if len(one_sent_list) > 2:
#        one_sent = ' '.join(one_sent_list)
#        sent_list.append(one_sent)
#        
#doc_vec = sent_embed(sent_list).numpy().tolist()


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
