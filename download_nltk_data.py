#!/usr/bin/env python3
"""
NLTK Data Downloader

This script downloads the required NLTK data for the palindrome generator.
"""

import nltk
import os

def main():
    print("Downloading required NLTK data...")
    
    # Download required NLTK datasets
    nltk.download('punkt')
    nltk.download('words')
    nltk.download('averaged_perceptron_tagger')
    
    print("NLTK data downloaded successfully!")

if __name__ == "__main__":
    main()
