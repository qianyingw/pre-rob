#!/bin/bash
for file in rob/data/npqip/PDFs/*.pdf; do
  pdftotext "$file" "rob/data/npqip/TXTs/$(basename "$file" .pdf).txt"
done