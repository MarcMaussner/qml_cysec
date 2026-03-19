#!/bin/bash
# Script to compile the fake_sim research paper

# Navigate to the script's directory
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR"

echo "--- [1/4] Cleaning old auxiliary files ---"
rm -f main.aux main.bbl main.bcf main.blg main.log main.out main.run.xml

echo "--- [2/4] Initial LaTeX compilation ---"
pdflatex -interaction=nonstopmode main.tex

echo "--- [3/4] Processing bibliography with Biber ---"
biber main

echo "--- [4/4] Final LaTeX compilation ---"
pdflatex -interaction=nonstopmode main.tex

if [ $? -eq 0 ]; then
    echo "--------------------------------------------------"
    echo "✅ Success! main.pdf has been successfully updated."
    echo "--------------------------------------------------"
else
    echo "--------------------------------------------------"
    echo "❌ Errors occurred during compilation."
    echo "Please check 'main.log' for details."
    echo "--------------------------------------------------"
    exit 1
fi
