#!/bin/bash
set -e
rm -rf data

# Create and change directory
mkdir data
cd data

# Download kaggle dataset.
kaggle competitions download -c cassava-leaf-disease-classification

# Unzip downloaded file.
unzip -q cassava-leaf-disease-classification.zip

# Delete file
rm -rf cassava-leaf-disease-classification.zip