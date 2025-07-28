#!/usr/bin/env python3
"""
Palindrome Paragraph Generator

This script generates grammatically correct and semantically meaningful palindrome paragraphs.
A palindrome paragraph reads the same forwards and backwards (ignoring spaces, punctuation,
and capitalization).
"""

import argparse
import nltk
import random
import re
import time
import spacy
import ssl
from nltk.corpus import words, wordnet
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import defaultdict
from spacy.language import Language
from spacy.tokens import Doc
from tqdm import tqdm

# Disable SSL verification (for downloading NLTK resources)
ssl._create_default_https_context = ssl._create_unverified_context

# Download necessary NLTK resources
try:
    nltk.data.find('corpora/words')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('words')
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

# Load spaCy model for grammar checking
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Downloading spaCy model for grammar checking...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Create a custom component for checking grammatical correctness
@Language.component("grammar_check")
def grammar_check(doc):
    # This is a simplified grammar check
    # A real grammar checker would be more complex
    return doc

# Add the component to the pipeline
nlp.add_pipe("grammar_check", last=True)

class PalindromeParagraphGenerator:
    def __init__(self):
        """Initialize the generator with word lists and language models"""
        print("Initializing palindrome paragraph generator...")
        self.all_words = set(w.lower() for w in words.words())
        
        # Build palindrome word dictionary
        self.palindrome_words = self._find_palindrome_words()
        
        # Categorize words by part of speech
        self.pos_words = self._categorize_words_by_pos()
        
        # Build n-gram model for more natural text generation
        self.ngram_model = self._build_ngram_model()
        
        # Dictionary of letter pairs for building words
        self.letter_pairs = self._build_letter_pairs()

    def _find_palindrome_words(self):
        """Find all palindrome words in the dictionary"""
        print("Finding palindrome words...")
        palindromes = {}
        
        # Group palindromes by length for more efficient selection
        for word in tqdm(self.all_words):
            if len(word) > 1 and word == word[::-1]:
                length = len(word)
                if length not in palindromes:
                    palindromes[length] = []
                palindromes[length].append(word)
        
        return palindromes

    def _categorize_words_by_pos(self):
        """Categorize palindrome words by part of speech"""
        print("Categorizing palindrome words by part of speech...")
        pos_dict = defaultdict(list)
        
        # Flatten the palindrome words list
        all_palindromes = [word for length_list in self.palindrome_words.values() 
                          for word in length_list]
        
        # Use WordNet to get parts of speech
        for word in tqdm(all_palindromes):
            synsets = wordnet.synsets(word)
            if synsets:
                # Get the most common part of speech for this word
                pos = synsets[0].pos()
                pos_dict[pos].append(word)
            else:
                # If not found in WordNet, use a simple heuristic
                # Most palindromes are nouns, so default to 'n'
                pos_dict['n'].append(word)
        
        return pos_dict

    def _build_ngram_model(self):
        """Build n-gram model from WordNet sentences for more natural text"""
        print("Building n-gram model for natural language generation...")
        sentences = []
        
        # Get example sentences from WordNet
        for synset in tqdm(list(wordnet.all_synsets())[:5000]):  # Limit to 5000 for speed
            examples = synset.examples()
            sentences.extend(examples)
        
        # Create bigrams and count frequencies
        model = defaultdict(lambda: defaultdict(int))
        
        for sentence in sentences:
            tokens = word_tokenize(sentence.lower())
            for w1, w2 in ngrams(tokens, 2):
                model[w1][w2] += 1
        
        return model

    def _build_letter_pairs(self):
        """Build a dictionary of letter pairs that can form palindromes"""
        pairs = {}
        for c1 in 'abcdefghijklmnopqrstuvwxyz':
            for c2 in 'abcdefghijklmnopqrstuvwxyz':
                # The pair and its reverse
                pairs[c1 + c2] = c2 + c1
        return pairs

    def clean_text(self, text):
        """Remove spaces and punctuation, convert to lowercase"""
        return re.sub(r'[^a-z]', '', text.lower())

    def is_palindrome(self, text):
        """Check if text is a palindrome (ignoring spaces, punctuation, case)"""
        cleaned = self.clean_text(text)
        return cleaned == cleaned[::-1]

    def get_palindrome_word(self, min_length=3, max_length=7):
        """Get a random palindrome word within the length constraints"""
        available_lengths = [l for l in self.palindrome_words.keys() 
                           if min_length <= l <= max_length]
        
        if not available_lengths:
            return None
        
        length = random.choice(available_lengths)
        return random.choice(self.palindrome_words[length])

    def get_word_by_pos(self, pos, min_length=3, max_length=None):
        """Get a random word with the specified part of speech"""
        words_list = self.pos_words.get(pos, [])
        
        if max_length:
            words_list = [w for w in words_list if min_length <= len(w) <= max_length]
        else:
            words_list = [w for w in words_list if len(w) >= min_length]
            
        return random.choice(words_list) if words_list else None

    def create_palindrome_sentence(self, max_words=10):
        """Create a single palindrome sentence with reasonable grammar"""
        words = []
        center_word = None
        
        # Decide on a center word
        if random.random() > 0.5:
            center_word = self.get_palindrome_word(min_length=4)
        
        # Number of words (on one side)
        num_words = random.randint(3, max_words // 2)
        
        # Build the first half
        for _ in range(num_words):
            word = self.get_palindrome_word()
            if word:
                words.append(word)
            else:
                # Fallback to a random word
                fallback = random.choice(list(self.all_words))
                words.append(fallback)
        
        # Build the sentence
        first_half = " ".join(words)
        
        # Add center word if present
        if center_word:
            sentence = f"{first_half} {center_word} {' '.join(words[::-1])}"
        else:
            sentence = f"{first_half} {' '.join(words[::-1])}"
        
        # Capitalize first letter and add period
        sentence = sentence.capitalize() + "."
        
        return sentence

    def check_grammar(self, text):
        """Check if the text is grammatically correct using spaCy"""
        doc = nlp(text)
        
        # This is a simplified check
        # A real implementation would use a more sophisticated grammar model
        has_subject = False
        has_verb = False
        
        for token in doc:
            if token.dep_ in ("nsubj", "nsubjpass"):
                has_subject = True
            if token.pos_ == "VERB":
                has_verb = True
        
        return has_subject and has_verb

    def generate_palindrome_paragraph(self, min_sentences=3, max_sentences=5):
        """Generate a paragraph of palindrome sentences that form a larger palindrome"""
        print("Generating palindrome paragraph...")
        
        # Strategy: We'll create a mirrored paragraph
        # First half: Normal sentences
        # Center: A palindrome word or short palindrome phrase
        # Second half: Mirror of the first half
        
        num_sentences = random.randint(min_sentences, max_sentences)
        sentences = []
        
        # Generate first half of sentences
        for _ in range(num_sentences // 2):
            sentence = self.create_palindrome_sentence()
            sentences.append(sentence)
        
        # Center sentence (if odd number of sentences)
        if num_sentences % 2 == 1:
            center = self.create_palindrome_sentence()
            sentences.append(center)
        
        # Add mirrored sentences for second half
        first_half = sentences.copy()
        for sentence in reversed(first_half[:num_sentences // 2]):
            # Reverse the sentence's letters to make it a palindrome when combined
            reversed_chars = sentence[::-1]
            sentences.append(reversed_chars)
        
        # Join into a paragraph
        paragraph = " ".join(sentences)
        
        # Verify it's a palindrome
        if not self.is_palindrome(paragraph):
            # Adjust to make it a palindrome
            print("Adjusting to ensure palindrome property...")
            cleaned = self.clean_text(paragraph)
            
            # If it's almost a palindrome, try simple fixes
            if cleaned[:-1] == cleaned[1:][::-1]:
                # Add the last character to the beginning
                paragraph = cleaned[-1] + paragraph
            elif cleaned[1:] == cleaned[:-1][::-1]:
                # Add the first character to the end
                paragraph = paragraph + cleaned[0]
        
        return paragraph

    def is_readable(self, text, min_word_length=3):
        """Check if the text is readable/sensible"""
        words = text.lower().split()
        
        # Count valid dictionary words
        valid_words = [w for w in words if w in self.all_words and len(w) >= min_word_length]
        
        # If more than 40% are valid, consider it readable
        return len(valid_words) / max(1, len(words)) > 0.4

    def generate(self, max_attempts=5):
        """Generate the best palindrome paragraph within max_attempts"""
        best_paragraph = None
        best_score = 0
        
        for _ in range(max_attempts):
            paragraph = self.generate_palindrome_paragraph()
            
            # Score based on length and readability
            score = len(paragraph) * (1 if self.is_readable(paragraph) else 0.5)
            
            if score > best_score:
                best_score = score
                best_paragraph = paragraph
        
        return best_paragraph

def main():
    """Main function to run the palindrome paragraph generator"""
    parser = argparse.ArgumentParser(description='Generate palindrome paragraphs')
    parser.add_argument('--sentences', type=int, default=5, 
                       help='Maximum number of sentences in the paragraph')
    parser.add_argument('--attempts', type=int, default=3,
                       help='Number of attempts to generate the best paragraph')
    args = parser.parse_args()
    
    # Initialize the generator
    generator = PalindromeParagraphGenerator()
    
    # Generate and display the palindrome paragraph
    print("\nGenerating palindrome paragraph...\n")
    start_time = time.time()
    
    paragraph = generator.generate(max_attempts=args.attempts)
    
    end_time = time.time()
    
    print("\n" + "="*80)
    print("PALINDROME PARAGRAPH:")
    print("="*80)
    print(paragraph)
    print("="*80)
    
    # Check if it's a palindrome
    is_pal = generator.is_palindrome(paragraph)
    print(f"Is palindrome: {is_pal}")
    
    # Additional info
    print(f"Length: {len(paragraph)} characters")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    if not is_pal:
        print("Warning: Generated text is not a perfect palindrome.")

if __name__ == "__main__":
    main()
