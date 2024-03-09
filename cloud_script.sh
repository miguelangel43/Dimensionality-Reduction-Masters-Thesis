#!/bin/bash

# Install Python 3.9.1
sudo apt update
sudo apt install python3 unzip git

# Clone the repo
git clone https://github.com/miguelangel43/Dimensionality-Reduction-Masters-Thesis.git
cd Dimensionality-Reduction-Masters-Thesis/

# Install Python modules listed in requirements.txt
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

# Download census income dataset
wget https://archive.ics.uci.edu/static/public/20/census+income.zip
mkdir datasets
unzip census+income.zip -d datasets/census_income

# Set datasets path
echo "DATAFOLDER_PATH: $(pwd)/datasets/" > config.yml

# Make directories
mkdir dim scores evalues corr

# Run pipeline
python3 cloud_main.py
