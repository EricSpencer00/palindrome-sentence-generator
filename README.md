# Palindrome Sentence Generator

A tool for generating grammatically correct and semantically meaningful palindrome paragraphs.

## Introduction

This tool generates palindromes that read the same forwards and backwards, ignoring spaces, punctuation, and capitalization. It uses advanced NLP techniques to ensure the text is both a valid palindrome and makes grammatical sense.

## Features

- Character-level palindrome generation
- Middle-outward expansion technique
- Two-direction LLM-based generation
- Different word boundaries between first and second half
- Smart word and phrase selection for readability
- Validation and scoring of palindrome quality
- Performance metrics and error logging

## Installation

1. Clone this repository
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download required NLTK and spaCy models (first run will download these automatically)

4. (Optional) For LLM-based generation:
   - Copy `.env.template` to `.env`
   - Add your OpenAI API key to the `.env` file

## Usage

Basic usage:

```bash
python palindrome_generator.py
```

With options:

```bash
python palindrome_generator.py --length 200 --verbose
```

Options:

- `--sentences`: Minimum number of sentences (default: 5)
- `--attempts`: Number of generation attempts (default: 10)
- `--length`: Target character length
- `--verbose`: Show detailed output
- `--center`: Optional center word/character to start with
- `--method`: Generation method ('traditional', 'middle-out', 'bidirectional', or 'llm')
- `--use-openai`: Use OpenAI API for LLM-based generation (requires API key)

## Default Method

The generator now defaults to the **bidirectional** method, which ensures both halves of the palindrome are valid English and character-level symmetry is maintained.

## Generation Methods

The generator supports multiple approaches:

1. **Traditional**: Creates palindrome sentences by mirroring words around a center.

2. **Middle-Out**: Starts with a center character or word and builds outward, ensuring character-level palindrome properties while maintaining different word boundaries between halves.

3. **LLM/Bidirectional**: Uses language models to generate both sides with better semantics:
   - Generates the right side using a language model
   - Uses the character-reversed right side as a prompt for generating the left side
   - Ensures both halves are valid English
   - Post-processes for punctuation/spacing symmetry

## Testing

Run the test suite:

```bash
python test_palindrome_generator.py
```