#!/bin/bash

# Stroke
for file in /media/mynewdrive/rob/data/stroke/PDFs/*.pdf; do
  pdftotext "$file" "/media/mynewdrive/rob/data/stroke/TXTs/$(basename "$file" .pdf).txt"
done

# NP
for file in /media/mynewdrive/rob/data/np/PDFs/*.pdf; do
  pdftotext "$file" "/media/mynewdrive/rob/data/np/TXTs/$(basename "$file" .pdf).txt"
done

# Psychosis
for file in /media/mynewdrive/rob/data/psycho/PDFs/*.pdf; do
  pdftotext "$file" "/media/mynewdrive/rob/data/psycho/TXTs/$(basename "$file" .pdf).txt"
done

# NPQIP
for file in /media/mynewdrive/rob/data/npqip/PDFs/*.pdf; do
  pdftotext "$file" "/media/mynewdrive/rob/data/npqip/TXTs/$(basename "$file" .pdf).txt"
done

# IICARus
for file in /media/mynewdrive/rob/data/iicarus/PDFs/*.pdf; do
  pdftotext "$file" "/media/mynewdrive/rob/data/iicarus/TXTs/$(basename "$file" .pdf).txt"
done