#!/bin/bash
set -e

# Download kaggle dataset.
kaggle competitions download -c cassava-leaf-disease-classification

# Unzip downloaded file.
unzip cassava-leaf-disease-classification.zip