#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert TEI.XMLs to TXTs
Created on Fri Dec 13 12:15:27 2019
@author: qwang
"""

import os
import xml.etree.ElementTree as ET
import re


#%%

#xmlpath = '/media/mynewdrive/rob/stroke_969.tei.xml'

def convert_single_xml(xmlpath):
    tree = ET.parse(xmlpath)  
    root = tree.getroot()  
    text_list = []
    
    # Extract main body
    for div in root.findall("{http://www.tei-c.org/ns/1.0}text/{http://www.tei-c.org/ns/1.0}body/{http://www.tei-c.org/ns/1.0}div"):
        for h in div.findall("{http://www.tei-c.org/ns/1.0}head"):
            if h.text:
                text_list.append(ET.tostring(h, encoding="utf-8", method="xml").decode("utf-8"))
                
        for p in div.findall("{http://www.tei-c.org/ns/1.0}p"):
            if p.text:
                text_list.append(ET.tostring(p, encoding="utf-8", method="xml").decode("utf-8"))

#    # Extract acknowledgments
#    for div in root.findall("{http://www.tei-c.org/ns/1.0}text/{http://www.tei-c.org/ns/1.0}back/{http://www.tei-c.org/ns/1.0}div"):
#        if div.attrib['type'] == 'acknowledgement':
#            text_list.append(ET.tostring(div, encoding="utf-8", method="xml").decode("utf-8"))            
                
    text = '\n'.join(text_list)          
    text = re.sub("[\<].*?[\>]", "", text)  # Remove node information
    text = re.sub(r"\s+", " ", text)
    text = text.lstrip()
        
    return text


                           
#%%
def convert_xmls(data_dir, data_name):
    
    tei_dir = os.path.join(data_dir, data_name, 'TEIs')
    txt_dir = os.path.join(data_dir, data_name, 'GROTXTs')
    if os.path.exists(txt_dir) == False:
        os.makedirs(txt_dir)  
    
    for filename in os.listdir(tei_dir):   
        xml_path = os.path.join(tei_dir, filename)    
        text = convert_single_xml(xml_path)
        txt_path = os.path.join(txt_dir, re.sub('.xml', '.txt', filename))
        
        with open(txt_path, 'w', encoding='utf-8') as fout:
            fout.write(text)
            
#%%
data_dir = '/media/mynewdrive/rob/data'
convert_xmls(data_dir, 'stroke')
convert_xmls(data_dir, 'np')
convert_xmls(data_dir, 'npqip')
convert_xmls(data_dir, 'iicarus')
convert_xmls(data_dir, 'psycho')

####
#xmlpath = '/media/mynewdrive/rob/stroke_1452.tei.xml'
#tree = ET.parse(xmlpath)  
#root = tree.getroot()  
#txt_list = []
#for div in root.findall("{http://www.tei-c.org/ns/1.0}text/{http://www.tei-c.org/ns/1.0}body/{http://www.tei-c.org/ns/1.0}div"):
#    for p in div.findall("{http://www.tei-c.org/ns/1.0}p"):
#        if p.text:
#            txt_list.append(p.text)
#txt = '\n'.join(txt_list)        
#
#
####
#xmlpath = '/media/mynewdrive/rob/stroke_1452.tei.xml'
#tree = ET.parse(xmlpath)  
#root = tree.getroot()  
#txt_list = []
#for div in root.findall("{http://www.tei-c.org/ns/1.0}text/{http://www.tei-c.org/ns/1.0}body/{http://www.tei-c.org/ns/1.0}div"):
#    for p in div.findall("{http://www.tei-c.org/ns/1.0}p"):
#        if p.text:
#            txt_list.append(ET.tostring(p, encoding="utf-8", method="xml").decode("utf-8"))
#txt = '\n'.join(txt_list)       


