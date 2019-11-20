#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 12:14:08 2019

@author: qwang
"""


import os
import pandas as pd
import numpy as np

# change working directory
wdir = '/home/qwang/rob/'
os.chdir(wdir)

from src.data_process.df2json import df2json

#%% Copy pdfs from S drive (local PC)
#    os.chdir('U:/Datastore/CMVM/scs/groups/DCN/TRIALDEV/CAMARADES/Qianying/rob')
#    
#    neuro = pd.read_csv("data/np/NP_RawFinal.csv", sep=',', engine="python")   
#    neuro = neuro.dropna(how='all')
#    
#    # Remove records without file links
#    neuro = neuro.dropna(subset=['DocumentLink', 'fileLink'])  # remove 48 records
#    neuro = neuro[-neuro["PubID"].isin([29781])]  # fileLink is invalid
#    # Remove duplicates
#    neuro = neuro.drop_duplicates(subset=['PubID'], keep='first')
#    
#    import shutil
#    filename = []
#    # Clean fileLink
#    neuro['fileLink'] = neuro['fileLink'].str.replace('S:/TRIALDEV/', 'U:/Datastore/CMVM/scs/groups/DCN/TRIALDEV/')
#    neuro['fileLink'] = neuro['fileLink'].str.replace('\\', "/")
#    for i, row in neuro.iterrows():
#        link = neuro.loc[i,'fileLink']
#        neuro.loc[i,'fileLink'] = link.split(";")[0].split("//cmvm")[0]
#        if os.path.exists(neuro.loc[i,'fileLink']) == False:
#            print(neuro.loc[i,'fileLink'])
#        else:
#            filename.append(os.path.basename(neuro.loc[i,'fileLink']))
#            shutil.copy2(neuro.loc[i,'fileLink'], 'data/np/npPDFs')
#        
#    
#    filename_unique = set(filename)
#    filename_duplicate = set([x for x in filename if filename.count(x) > 1])

    
#%% Read and format
neuro = pd.read_csv("data/np/NP_RawFinal.csv", sep=',', engine="python")   
neuro = neuro.dropna(how='all')

# Remove records without file links
neuro = neuro.dropna(subset=['DocumentLink', 'fileLink'])  # remove 48 records
neuro = neuro[-neuro["PubID"].isin([29781, 24323])]  # fileLink is invalid; 24323 has same link as 68550
# Remove duplicates

neuro = neuro.drop_duplicates(subset=['PubID'], keep='first')

list(neuro.columns)

#    ['PubID',
#     'Surnames',
#     'Years',
#     'PublicationIDs',
#     'Randomisation',
#     'AllocationConcealment',
#     'BlindedOutcomeAssessment',
#     'SampleSizeCalculation',
#     'AnimalWelfareRegulations',
#     'ConflictsOfInterest',
#     'AnimalExclusions',
#     'DocumentLink',
#     'fileLink',
#     'source']

# Change column names
neuro.columns = ['PubID', 'Surnames', 'Years', 'PublicationIDs',
               
                 'RandomizationTreatmentControl',
                 'AllocationConcealment',
                 'BlindedOutcomeAssessment',
                 'SampleSizeCalculation',
                 'AnimalWelfareRegulations',
                 'ConflictsOfInterest',
                 'AnimalExclusions',
                 
                 'DocumentLink', 'fileLink', 'source'] 

# Add ID column
neuro['ID'] = np.arange(1, len(neuro)+1)  # len(neuro) = 1761


# Manual correction
neuro.loc[neuro.DocumentLink=='http:/www.degruyter.com/downloadpdf/j/tnsci.2015.6.issue-1/tnsci-2015-0010/tnsci-2015-0010.xml', 'DocumentLink'] = 'np/PDFs/tnsci-2015-0010.pdf'
neuro.loc[neuro.DocumentLink=='http:/www.rehab.research.va.gov/jour/09/46/1/Tan.html', 'DocumentLink'] = 'np/PDFs/Tan.pdf'
neuro.loc[neuro.DocumentLink=='Publications/NP_references/4368_Palazzo.htm', 'DocumentLink'] = 'np/PDFs/4368_Palazzo.pdf'


# Clean DocumentLink
neuro['DocumentLink'] = neuro['DocumentLink'].str.replace('\\', "/")
for i, row in neuro.iterrows():
    link = neuro.loc[i,'DocumentLink']
    neuro.loc[i,'DocumentLink'] = link.split(";")[0].split("//cmvm")[0]
    


# Modify DocumentLink
neuro['DocumentLink'] = neuro['DocumentLink'].str.replace('Publications/', 'np/PDFs/')

# Modify fileLink
pdf_folder = '/home/qwang/rob/data/'
neuro['fileLink'] = pdf_folder + neuro['DocumentLink'].astype(str)


#%% Convert pdf to txt by Xpdf
# Generate txt file for bash script
with open('data/np/np_doclink.txt', 'w') as fout:
    for link in neuro['DocumentLink']:     
        fout.write(link + '\n')
                

    

# See pdf2txt_np.sh





