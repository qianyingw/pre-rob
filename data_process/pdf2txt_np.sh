#!/bin/bash
cd /home/qwang/rob/data

while read -r line; do 
  pdftotext "$line" "np/TXTs/$(basename "$line" .pdf).txt"
  #arr+=("$line"); 
done<np/np_doclink.txt