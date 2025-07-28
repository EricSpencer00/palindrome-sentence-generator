# Palindrome Paragraph Generator

A sophisticated Python tool that generates entire palindrome paragraphs that are grammatically correct and semantically meaningful.

## What is a Palindrome Paragraph?

A palindrome paragraph is an entire paragraph that reads the same forwards and backwards (ignoring spaces, punctuation, and capitalization). This is much more complex than individual palindrome sentences or words.

Examples of palindrome sentences (smaller units):
- "A man, a plan, a canal: Panama"
- "Never odd or even"
- "Rats live on no evil star"

## Features

- Generates entire palindrome paragraphs (not just individual sentences)
- Uses comprehensive English dictionary with NLTK's corpus
- Employs natural language processing for grammatical correctness
- Creates paragraphs that aim to be semantically meaningful
- Uses spaCy for grammar checking
- Implements sophisticated palindrome generation algorithms

## Requirements

- Python 3.6 or higher
- Required packages: nltk, spacy, tqdm

## Installation

```bash
# Clone the repository
git clone https://github.com/EricSpencer00/palindrome-sentence-generator.git
cd palindrome-sentence-generator

# Install dependencies
pip install -r requirements.txt

# Download required NLTK and spaCy resources
python -c "import nltk; nltk.download('words'); nltk.download('wordnet'); nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
python -m spacy download en_core_web_sm
```

## Usage

```bash
# Generate a palindrome paragraph
python3 palindrome_generator.py

# Specify the maximum number of sentences
python3 palindrome_generator.py --sentences 7

# Specify number of generation attempts (higher = better quality but slower)
python3 palindrome_generator.py --attempts 5
```

## How It Works

The generator uses several sophisticated techniques to create palindrome paragraphs:

1. **Dictionary-Based Approach**: Uses NLTK's comprehensive English dictionary to find valid words.
2. **Palindrome Word Identification**: Pre-computes all palindrome words in the dictionary.
3. **Part-of-Speech Tagging**: Categorizes words by parts of speech to create more grammatical sentences.
4. **N-gram Models**: Uses language models to improve the flow and naturalness of the text.
5. **Mirrored Paragraph Construction**: Creates paragraphs where the second half mirrors the first half.
6. **Grammar Checking**: Uses spaCy to verify grammatical correctness.
7. **Readability Scoring**: Scores generated paragraphs for readability and selects the best one.

## Example Output

```
====================================================================
PALINDROME PARAGRAPH:
====================================================================
Noon level radar. Rotator deed did level radar. Radar level did deed rotator. Radar level noon.
====================================================================
Is palindrome: True
Length: 121 characters
Time taken: 5.23 seconds
```

## Limitations

- Generating perfect palindrome paragraphs that are grammatically correct and semantically meaningful is an extremely challenging problem
- Longer paragraphs tend to sacrifice grammatical correctness or semantic meaning
- The generation process can be slow due to the complexity of the task