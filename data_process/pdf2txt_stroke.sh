#!/bin/bash
for file in /media/mynewdrive/rob/data/stroke/PDFs/*.pdf; do
  pdftotext "$file" "/media/mynewdrive/rob/data/stroke/TXTs/$(basename "$file" .pdf).txt"
done