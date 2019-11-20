#!/bin/bash
for file in rob/data/psycho/PDFs/*.pdf; do
  pdftotext "$file" "rob/data/psycho/TXTs/$(basename "$file" .pdf).txt"
done

