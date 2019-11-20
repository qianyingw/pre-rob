#!/bin/bash
for file in rob/data/iicarus/PDFs/*.pdf; do
  pdftotext "$file" "rob/data/iicarus/TXTs/$(basename "$file" .pdf).txt"
done