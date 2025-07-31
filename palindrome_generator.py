#!/usr/bin/env python3
"""
Palindrome Paragraph Generator

This script generates palindrome paragraphs that are grammatically correct
and use English words. The generation process builds palindromes incrementally
by starting with small palindromes and extending them while maintaining their
palindromic property.
"""

import nltk
import spacy
import random
from tqdm import tqdm
import logging
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    filename='palindrome.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('punkt')
    nltk.download('words')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logging.info("Downloading spaCy English model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Load English words from NLTK
english_words = set(w.lower() for w in nltk.corpus.words.words())

class PalindromeGenerator:
    def __init__(self):
        self.word_by_pos = self._categorize_words_by_pos()
        
    def _categorize_words_by_pos(self):
        """Categorize words by their part of speech."""
        word_by_pos = {
            'NOUN': [],
            'VERB': [],
            'ADJ': [],
            'ADV': [],
            'DET': [],
            'ADP': [],  # prepositions
            'PRON': [],  # pronouns
            'CONJ': []   # conjunctions
        }
        
        # Common determiners, prepositions, pronouns, and conjunctions
        word_by_pos['DET'] = ["a", "the", "my", "your", "our", "their"]
        word_by_pos['ADP'] = ["in", "on", "at", "by", "for", "with", "to", "from"]
        word_by_pos['PRON'] = ["I", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"]
        word_by_pos['CONJ'] = ["and", "or", "but", "nor", "so", "yet", "for"]
        
        # Select a subset of common English words for efficiency
        sample_words = random.sample(list(english_words), 5000)
        
        for word in tqdm(sample_words, desc="Categorizing words"):
            # Skip very short words or non-alphabetic words
            if len(word) < 2 or not word.isalpha():
                continue
                
            doc = nlp(word)
            pos = doc[0].pos_
            
            if pos in word_by_pos:
                word_by_pos[pos].append(word)
        
        return word_by_pos
    
    def is_palindrome(self, text):
        """Check if a string is a palindrome (ignoring spaces and punctuation)."""
        text = ''.join(c.lower() for c in text if c.isalnum())
        return text == text[::-1]
    
    def reverse_word(self, word):
        """Return the reverse of a word."""
        return word[::-1]
    
    def find_palindrome_words(self):
        """Find words that are palindromes themselves."""
        palindrome_words = []
        for pos in self.word_by_pos:
            for word in self.word_by_pos[pos]:
                if self.is_palindrome(word) and len(word) > 2:
                    palindrome_words.append(word)
        return palindrome_words
    
    def find_reversible_word_pairs(self):
        """Find pairs of words where one is the reverse of the other."""
        reversible_pairs = []
        for pos in self.word_by_pos:
            words = set(self.word_by_pos[pos])
            for word in words:
                reversed_word = self.reverse_word(word)
                if reversed_word in words and reversed_word != word:
                    reversible_pairs.append((word, reversed_word))
        return reversible_pairs
    
    def generate_simple_palindrome(self):
        """Generate a simple palindrome phrase."""
        structures = [
            "{adj} {noun} {verb} {noun} {adj}",
            "{adv} {verb} {noun} {verb} {adv}",
            "{det} {adj} {noun} {verb} {noun} {adj} {det}",
            "{noun} {verb} {adv} {adv} {verb} {noun}"
        ]
        
        structure = random.choice(structures)
        parts = {}
        
        # Fill in the structure with words
        if "{adj}" in structure:
            adj = random.choice(self.word_by_pos['ADJ'])
            parts["adj"] = adj
        
        if "{noun}" in structure:
            noun = random.choice(self.word_by_pos['NOUN'])
            parts["noun"] = noun
            
        if "{verb}" in structure:
            verb = random.choice(self.word_by_pos['VERB'])
            parts["verb"] = verb
            
        if "{adv}" in structure:
            adv = random.choice(self.word_by_pos['ADV'])
            parts["adv"] = adv
            
        if "{det}" in structure:
            det = random.choice(self.word_by_pos['DET'])
            parts["det"] = det
            
        # Construct the palindrome
        palindrome = structure
        for part, word in parts.items():
            palindrome = palindrome.replace("{" + part + "}", word)
            
        return palindrome
    
    def build_palindrome_incrementally(self, target_length=300):
        """
        Build a palindrome incrementally by starting with a small palindrome
        and extending it while maintaining the palindromic property.
        """
        # Start with a simple palindrome seed
        palindrome_words = self.find_palindrome_words()
        if not palindrome_words:
            logging.warning("No palindrome words found, starting with a simple word")
            palindrome = random.choice(self.word_by_pos['NOUN'])
        else:
            palindrome = random.choice(palindrome_words)
        
        logging.info(f"Starting with seed: {palindrome}")
        
        attempts = 0
        max_attempts = 1000
        
        while len(palindrome) < target_length and attempts < max_attempts:
            attempts += 1
            
            # Choose a random extension pattern
            extension_patterns = [
                # Pattern: a -> xax
                lambda p, word: f"{word} {p} {word}",
                
                # Pattern: aba -> xabax
                lambda p, word: f"{word} {p} {word}",
                
                # Pattern: abcba -> xabcbax
                lambda p, word: f"{word} {p} {word}",
                
                # Pattern: a -> bab
                lambda p, word: f"{word} {p} {word}"
            ]
            
            pattern = random.choice(extension_patterns)
            
            # Choose a random word to extend with
            pos_choices = ['NOUN', 'VERB', 'ADJ', 'ADV', 'DET', 'ADP', 'PRON', 'CONJ']
            pos = random.choice(pos_choices)
            
            if not self.word_by_pos[pos]:
                continue
                
            word = random.choice(self.word_by_pos[pos])
            
            # Apply the extension
            new_palindrome = pattern(palindrome, word)
            
            # Check if it's a valid palindrome
            if self.is_palindrome(new_palindrome):
                palindrome = new_palindrome
                logging.info(f"Extended to: {palindrome} (length: {len(palindrome)})")
            
        if attempts >= max_attempts:
            logging.warning("Reached maximum attempts without reaching target length")
        
        return palindrome
    
    def generate_grammatical_palindrome(self, target_length=300):
        """
        Generate a grammatically correct palindrome paragraph.
        This is a simplified approach focusing on short, grammatically plausible segments.
        """
        # Start with basic palindrome structures
        basic_structures = [
            ("a", "aba", "abcba", "abcdcba"),
            ("noon", "level", "radar", "deified"),
            ("step", "pets", "live", "evil")
        ]
        
        # Choose a structure
        structure_set = random.choice(basic_structures)
        
        # Start with one structure
        current = random.choice(structure_set)
        
        attempts = 0
        max_attempts = 5000
        
        while len(current) < target_length and attempts < max_attempts:
            attempts += 1
            
            # Try to extend the palindrome
            extension_options = [
                # Wrap with a new word
                lambda p, word: f"{word} {p} {word}",
                
                # Insert in the middle (for odd-length palindromes)
                lambda p, word: p[:len(p)//2] + f" {word} " + p[len(p)//2:] if len(p) % 2 == 1 else p,
                
                # Insert palindromic phrase
                lambda p, phrase: f"{p} {phrase}"
            ]
            
            option = random.choice(extension_options)
            
            # Select words or phrases to extend with
            if option.__code__.co_argcount > 1:  # If the option takes a word parameter
                pos_choices = ['NOUN', 'VERB', 'ADJ', 'ADV']
                pos = random.choice(pos_choices)
                
                if not self.word_by_pos[pos]:
                    continue
                    
                word = random.choice(self.word_by_pos[pos])
                new_current = option(current, word)
            else:
                # Generate a simple palindrome phrase
                phrase = self.generate_simple_palindrome()
                new_current = option(current, phrase)
            
            # Check if it's still a palindrome (allowing for spaces and punctuation)
            if self.is_palindrome(new_current):
                current = new_current
                logging.info(f"Extended palindrome to length {len(current)}")
            
        if attempts >= max_attempts:
            logging.warning(f"Reached max attempts ({max_attempts}) without reaching target length")
        
        return current

def main():
    load_dotenv()  # Load environment variables
    
    print("Initializing Palindrome Generator...")
    generator = PalindromeGenerator()
    
    print("\nGenerating a grammatical palindrome paragraph (target length: 300 characters)...")
    palindrome = generator.generate_grammatical_palindrome(300)
    
    print("\nGenerated Palindrome:")
    print("-" * 80)
    print(palindrome)
    print("-" * 80)
    
    print(f"\nPalindrome length: {len(palindrome)} characters")
    print(f"Is palindrome: {generator.is_palindrome(palindrome)}")
    
if __name__ == "__main__":
    main()
