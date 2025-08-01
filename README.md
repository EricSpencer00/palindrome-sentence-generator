# Palindrome Sentence Generator

A tool for generating grammatically correct and semantically meaningful palindrome paragraphs.

## Introduction

This tool generates palindromes that read the same forwards and backwards, ignoring spaces, punctuation, and capitalization. It uses advanced NLP techniques to ensure the text is both a valid palindrome and makes grammatical sense.

## Features

- Character-level palindrome generation that creates a single palindrome (not multiple separate palindromes)
- Multiple generation strategies:
  - LLM-based generation with optimized prompts
  - Constructive generation using algorithmic techniques
  - Improved multi-sentence generation
- Constructive approaches:
  - Center-out expansion technique
  - Mirror-based palindrome construction
  - Combinatorial palindrome generation
  - Famous palindrome pattern integration
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
python main.py --min-length 80 --method constructive
```

Options:

- `--min-length`: Minimum character length when normalized (default: 60)
- `--attempts`: Number of generation attempts (default: 5)
- `--use-nltk-seed`: Use NLTK to generate a palindrome seed
- `--model`: Model to use for LLM-based generation (default: "google/gemma-3n-e2b-it:free")
- `--method`: Generation method to use - "llm", "constructive", or "improved" (default: "llm")
- `--enhanced`: Use enhanced generation algorithm (default: True)
- `--use-improved`: Use the improved multi-sentence generator

## Advanced Usage

### Using the Constructive Generator

The constructive generator builds palindromes using algorithmic approaches without requiring an API key:

```bash
python main.py --method=constructive --min-length=80
```

See [CONSTRUCTIVE_APPROACH.md](CONSTRUCTIVE_APPROACH.md) for details on this method.

### Using the Improved Generator

The improved generator creates multi-sentence palindromic text with better grammar and coherence:

```bash
python main.py --method=improved --min-length=80 --attempts=10
```

You can also run the improved generator directly:

```bash
python improved_generator.py --min-length=80 --attempts=10
```

### Fallback Generator

If LLM-based generation fails, the program will automatically use a local fallback generator:

```bash
python fallback_generator.py --min-length=60
```
- `--output`: Output file to save the generated palindrome
- `--verbose`: Show detailed output and timing information
- `--improve-grammar`: Attempt to improve grammar of the generated palindrome

### Improved Generator

For best results, use the improved generator:

```bash
python improved_generator.py --length=250 --attempts=10 --improve-attempts=15 --verbose
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

The generator now supports three main approaches:

### 1. LLM-based Generation
Uses language models to generate palindromes with strong semantic coherence:
- Generates text using a language model with specialized prompts
- Uses NLTK seeds for better starting points
- Ensures character-level palindrome properties
- Requires an API key (OpenAI or compatible service)

### 2. Constructive Generation
Uses algorithmic approaches to build palindromes with specific structures:
- **Center-out**: Starts with a center word/phrase and builds outward symmetrically
- **Mirrored**: Creates first half then mirrors it to form a perfect palindrome
- **Combinatorial**: Combines multiple smaller palindromes to create longer ones
- Uses famous palindrome patterns and word pairs that form palindromes
- Does not require an API key

### 3. Improved Multi-sentence Generation
Creates more complex palindromes with multiple sentences:
- Uses grammar-based templates for better sentence structure
- Employs language models for semantic improvement
- Balances grammar quality and length requirements
- Features aggressive expansion for longer palindromes
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
## Examples

Here are examples of palindromes generated with different methods:

### Classic Palindromes
```
Madam, I'm Adam.
```

```
A man, a plan, a canal: Panama!
```

```
Rats live on no evil star.
```

```
Never odd or even.
```

### Constructive Generation Examples
```
V x f w g y llu. L d d r prod ara st o. Ava p b brod apa. Ata aba en e aba. Ata apa dor b b p. Ava ots ara dorp. R d d lull. Y g w f x. V.
```

```
Knot kapa bussu p b prat nay dw apa ta k. Feer hon os o. O. Loo son oh. Reef kat apa w d yan tarp b puss u bap ak tonk.
```

### LLM-based Generation Examples
```
Race fast, safe car! No, it is opposition. No, it is a sign of raw war. Fongs, I sit, I, on, no is opposition, no, it is opposition. No, it is a sign of raw war. Fongs, I sit, I, on, no is opposition, no, it is a fast car.
```

### Multi-sentence Examples
Run the generator with `--method=improved` to see longer, multi-sentence palindromes.

## Recent Improvements

### Constructive Generator
- Added four different constructive generation strategies
- Implemented famous palindrome pattern integration
- Enhanced word pair detection for better palindrome structure
- Improved text conversion for more readable output

### Improved Generator
- Implemented multi-attempt generation with best candidate selection
- Enhanced grammar improvement algorithm with aggressive strategies
- Added weighted scoring to balance grammar quality and length requirements

### Main Program
- Added support for multiple generation methods
- Improved method selection and fallback mechanisms
- Enhanced verbose output for debugging and analysis

### Testing
- Added test functions for constructive generation
- Enhanced validation of all components