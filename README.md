# Palindrome Sentence Generator

A tool for generating grammatically correct and semantically meaningful palindrome paragraphs.

## Introduction

This tool generates palindromes that read the same forwards and backwards, ignoring spaces, punctuation, and capitalization. It uses advanced NLP techniques to ensure the text is both a valid palindrome and makes grammatical sense.

## Features

- Character-level palindrome generation that creates a single palindrome (not multiple separate palindromes)
- Middle-outward expansion technique
- LLM-based generation with optimized prompts
- Structured sentence generation for better grammar
- Multiple generation strategies with fallbacks
- Smart word and phrase selection for readability
- Automatic repair of near-palindromes
- Validation and scoring of palindrome quality
- Multi-sentence support for more coherent text
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
python main.py
```

With options:

```bash
python main.py --length 200 --method grammar --verbose
```

Options:

- `--min-length`: Minimum character length when normalized (default: 60)
- `--attempts`: Number of generation attempts (default: 5)
- `--use-nltk-seed`: Use NLTK to generate a palindrome seed
- `--model`: Model to use for LLM-based generation (default: "google/gemma-3n-e2b-it:free")
- `--enhanced`: Use enhanced generation algorithm (default: True)
- `--use-improved`: Use the improved multi-sentence generator

## Advanced Usage

### Using the Improved Generator

The improved generator creates multi-sentence palindromic text with better grammar and coherence:

```bash
python main.py --use-improved --min-length 80 --attempts 10
```

You can also run the improved generator directly:

```bash
python improved_generator.py --min-length 80 --attempts 10
```

### Fallback Generator

If LLM-based generation fails, the program will automatically use a local fallback generator:

```bash
python fallback_generator.py --min-length 60
```
- `--output`: Output file to save the generated palindrome
- `--verbose`: Show detailed output and timing information
- `--improve-grammar`: Attempt to improve grammar of the generated palindrome

### Improved Generator

For best results, use the improved generator:

```bash
python improved_generator.py --length 250 --attempts 10 --improve-attempts 15 --verbose
```

Advanced options (improved generator):

- `--length`: Target character length (default: 250)
- `--attempts`: Number of generation attempts (default: 10)
- `--improve-attempts`: Number of grammar improvement attempts (default: 15)
- `--parallel`: Use parallel generation for better results
- `--threads`: Number of parallel threads (default: 4)
- `--output`: Output file to save the generated palindrome
- `--verbose`: Show detailed output and timing information

For advanced usage with the original methods:

```bash
python palindrome_generator.py --length 200 --verbose
```

Advanced options (original generator):

- `--sentences`: Minimum number of sentences (default: 5)
- `--attempts`: Number of generation attempts (default: 10)
- `--center`: Optional center word/character to start with
- `--method`: Generation method ('traditional', 'middle-out', 'bidirectional', or 'llm')
- `--use-openai`: Use OpenAI API for LLM-based generation (requires API key)

## Default Method

The generator now defaults to the **bidirectional** method, which ensures both halves of the palindrome are valid English and character-level symmetry is maintained.

## Generation Methods

The generator supports multiple approaches:

1. **Basic**: Creates palindrome sentences by mirroring words and building incrementally, focusing on grammatical correctness.

2. **Grammar-based**: Uses grammar rules to ensure the palindrome follows English syntax and is semantically meaningful.

3. **Traditional**: Creates palindrome sentences by mirroring words around a center.

4. **Middle-Out**: Starts with a center character or word and builds outward, ensuring character-level palindrome properties while maintaining different word boundaries between halves.

5. **LLM/Bidirectional**: Uses language models to generate both sides with better semantics:
   - Generates the right side using a language model
   - Uses the character-reversed right side as a prompt for generating the left side
   - Ensures both halves are valid English
   - Post-processes for punctuation/spacing symmetry

## Testing

Run the test suite:

```bash
python test_palindrome_generator.py
```

## Recent Improvements

### Improved Generator
- Added parallel generation capability for better results
- Implemented multi-attempt generation with best candidate selection
- Enhanced grammar improvement algorithm with aggressive strategies
- Added weighted scoring to balance grammar quality and length requirements

### Grammar Palindrome Generator
- Enhanced seed options for better starting points
- Improved wrapping templates for aggressive expansion
- Added timeout mechanisms to prevent generation loops
- Optimized expansion strategies for different target lengths

### Grammar Validator
- Improved grammar scoring algorithm
- Added more strategies for grammar improvement
- Enhanced suggestion generation for better readability

### Main Program
- Increased grammar improvement attempts
- Better tracking of generation progress
- Enhanced verbose output for debugging and analysis

### Testing
- Added `test_improvements.py` for focused testing of new features
- Enhanced `test_final.py` for comprehensive validation of all components