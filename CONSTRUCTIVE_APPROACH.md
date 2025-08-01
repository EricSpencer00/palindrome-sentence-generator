# Palindrome Sentence Generator

This project implements multiple approaches to generating palindromic sentences, with a focus on creating natural-sounding text that is also a character-level palindrome.

## Generation Methods

The project now includes three main generation methods:

1. **LLM-based Generation**: Uses language models to create palindromes based on prompts and seeds.
2. **Improved Multi-sentence Generator**: Creates more complex palindromes with multiple sentences.
3. **Constructive Generator**: Uses constructive approaches inspired by academic research to build palindromes with specific structures.

## Constructive Palindrome Generation

The constructive generator uses four main strategies:

### 1. Constructive Strategy
Builds palindromes by combining smaller palindromic units and extending them outward. This creates palindromes with varied structure.

### 2. Center-out Strategy
Starts with a center word or letter and builds the palindrome outward symmetrically. This creates well-structured palindromes with a clear central point.

Features:
- Uses palindrome words as centers
- Incorporates famous palindrome phrases
- Expands using word pairs, palindromic sequences, and common patterns

### 3. Mirrored Strategy
Creates the first half of the palindrome and then mirrors it to form a perfect palindrome. This tends to create highly symmetric structures.

### 4. Combinatorial Strategy
Combines multiple smaller palindromes to create a longer, more varied palindrome. This can produce more complex structures.

## Key Techniques

The implementation includes several advanced techniques:

1. **Word Pairs**: Uses pairs of words that form palindromes when combined (e.g., "live"/"evil", "draw"/"ward")
2. **Palindromic Seeds**: Starts with known palindrome words or phrases
3. **Structural Patterns**: Implements common palindromic structures and patterns
4. **Text Conversion**: Transforms normalized palindromes into readable text with proper spacing and punctuation

## Usage

To use the constructive palindrome generator:

```bash
python main.py --method=constructive --min-length=80
```

Additional options:
- `--attempts`: Number of generation attempts (default: 5)
- `--min-length`: Minimum palindrome length (default: 60)
- `--method`: Generation method to use - "llm", "constructive", or "improved" (default: "llm")

## Example Output

```
Generated text: Madam I'm Adam. A man a plan a canal Panama.
Normalized length: 31 characters
Normalized text: madamimamanamaplanacanalpanama
```

## References

The constructive approach is inspired by research on algorithmic palindrome construction and methods for creating "the most complete palindromic sentence" that contains as many words as possible from a dictionary.
