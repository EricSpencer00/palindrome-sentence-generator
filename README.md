# Palindrome Sentence Generator

A simple Python tool that generates palindrome sentences that are grammatically correct or make sense.

## What is a Palindrome?

A palindrome is a word, phrase, number, or other sequence of characters that reads the same forward and backward (ignoring spaces, punctuation, and capitalization).

Examples:
- "radar"
- "A man, a plan, a canal: Panama"
- "Never odd or even"

## Features

- Generates palindrome sentences using different methods
- Attempts to create grammatically correct or sensible palindromes
- Simple command-line interface

## Usage

```bash
# Generate a random palindrome from predefined phrases
python3 palindrome_generator.py

# Generate a mirrored palindrome (word-based reflection)
python3 palindrome_generator.py --method mirrored

# Generate a structured palindrome (attempts better grammar)
python3 palindrome_generator.py --method structured

# Generate multiple palindromes
python3 palindrome_generator.py --count 5
```

## Methods

1. **Random**: Selects from predefined, grammatically correct palindrome phrases
2. **Mirrored**: Creates a palindrome by mirroring palindromic words around a center
3. **Structured**: Builds palindromes with more structured grammar patterns

## Requirements

- Python 3.6 or higher

## Installation

No installation required. Simply download the script and run it with Python.

```bash
# Make the script executable (optional)
chmod +x palindrome_generator.py

# Run directly
./palindrome_generator.py
```