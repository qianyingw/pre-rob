#!/bin/bash
for file in /media/mynewdrive/rob/data/np/PDFs/*.pdf; do
  pdftotext "$file" "/media/mynewdrive/rob/data/np/TXTs/$(basename "$file" .pdf).txt"
done