#!/usr/bin/env bash
# exit on error immediately
set -o errexit

echo "--- Installing Python Dependencies ---"
pip install -r requirements.txt

echo "--- Downloading spaCy Model ---"
# This downloads the model into the environment where the app will run
python -m spacy download en_core_web_sm

echo "--- Build Complete ---"