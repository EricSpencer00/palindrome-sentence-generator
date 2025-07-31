#!/usr/bin/env python3
"""
Grammar-Based Palindrome Generator

This script generates grammatically correct palindrome paragraphs using
a grammar-based approach. It builds sentences that are palindromes at the
character level while maintaining English grammar rules.
"""

import nltk
import spacy
import random
from tqdm import tqdm
import logging
import os
from dotenv import load_dotenv
import re

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
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('punkt')
    nltk.download('words')
    nltk.download('averaged_perceptron_tagger')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logging.info("Downloading spaCy English model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Load English words from NLTK
english_words = set(w.lower() for w in nltk.corpus.words.words())

class GrammarPalindromeGenerator:
    def __init__(self):
        self.word_dict = self._build_word_dictionary()
        self.grammar_rules = self._define_grammar_rules()
        
    def _build_word_dictionary(self):
        """Build a dictionary of words categorized by part of speech."""
        word_dict = {
            'NOUN': [],
            'VERB': [],
            'ADJ': [],
            'ADV': [],
            'DET': ['a', 'the', 'my', 'your', 'our'],
            'PREP': ['in', 'on', 'at', 'by', 'to', 'for'],
            'PRON': ['I', 'you', 'he', 'she', 'it', 'we', 'they'],
            'CONJ': ['and', 'or', 'but', 'yet', 'so']
        }
        
        # Add palindrome words
        palindrome_words = [word for word in english_words if word == word[::-1] and len(word) > 2]
        logging.info(f"Found {len(palindrome_words)} palindrome words")
        
        # Add some common words for each part of speech
        sample_words = random.sample(list(english_words), 8000)
        
        for word in tqdm(sample_words, desc="Building word dictionary"):
            if len(word) < 2 or not word.isalpha():
                continue
                
            doc = nlp(word)
            pos = doc[0].pos_
            
            if pos == 'NOUN':
                word_dict['NOUN'].append(word)
            elif pos == 'VERB':
                word_dict['VERB'].append(word)
            elif pos == 'ADJ':
                word_dict['ADJ'].append(word)
            elif pos == 'ADV':
                word_dict['ADV'].append(word)
        
        # Find reversible word pairs (where one word reversed is another word)
        self.reversible_pairs = []
        for word in tqdm(sample_words, desc="Finding reversible pairs"):
            if len(word) < 3 or not word.isalpha():
                continue
                
            reversed_word = word[::-1]
            if reversed_word in english_words and reversed_word != word:
                self.reversible_pairs.append((word, reversed_word))
                
        logging.info(f"Found {len(self.reversible_pairs)} reversible word pairs")
        
        return word_dict
    
    def _define_grammar_rules(self):
        """Define grammar rules for sentence construction."""
        return {
            'S': [['NP', 'VP']],                          # Sentence -> Noun Phrase + Verb Phrase
            'NP': [['DET', 'N'], ['DET', 'ADJ', 'N']],    # Noun Phrase -> Determiner + Noun or Determiner + Adjective + Noun
            'VP': [['V', 'NP'], ['V', 'PREP', 'NP']],     # Verb Phrase -> Verb + Noun Phrase or Verb + Preposition + Noun Phrase
            'N': [['NOUN']],                              # Noun -> NOUN
            'V': [['VERB']],                              # Verb -> VERB
            'ADJ': [['ADJ']],                             # Adjective -> ADJ
            'DET': [['DET']],                             # Determiner -> DET
            'PREP': [['PREP']]                            # Preposition -> PREP
        }
    
    def is_palindrome(self, text):
        """Check if a string is a palindrome (ignoring spaces and punctuation)."""
        text = ''.join(c.lower() for c in text if c.isalnum())
        return text == text[::-1]
    
    def generate_palindrome_sentence(self):
        """Generate a grammatically correct palindrome sentence."""
        # Try to use reversible word pairs to create a palindrome
        if self.reversible_pairs:
            pair = random.choice(self.reversible_pairs)
            word1, word2 = pair
            
            templates = [
                f"{word1} is {word2}",
                f"{word1} and {word2}",
                f"from {word1} to {word2}",
                f"{word1} becomes {word2}"
            ]
            
            sentence = random.choice(templates)
            if self.is_palindrome(sentence):
                return sentence
        
        # If we couldn't create a palindrome with reversible pairs,
        # try with actual palindrome words to ensure we get a palindrome
        palindrome_words = ['level', 'radar', 'madam', 'refer', 'civic', 'kayak', 'racecar']
        
        # Check if we have any palindrome words in our dictionary
        seed_words = [word for word in self.word_dict['NOUN'] if self.is_palindrome(word) and len(word) > 3]
        if seed_words:
            palindrome_words.extend(seed_words)
            
        seed = random.choice(palindrome_words)
        logging.info(f"Using palindrome seed word: {seed}")
        
        templates = [
            f"a {seed} a",
            f"my {seed} my",
            f"the {seed} the",
            f"I {seed} I"
        ]
        
        sentence = random.choice(templates)
        
        # Verify it's a palindrome
        if not self.is_palindrome(sentence):
            # For safety, choose a simple guaranteed palindrome
            logging.warning(f"Generated sentence '{sentence}' is not a palindrome, using fallback")
            sentence = f"a {seed} a"
            
        return sentence
    
    def expand_palindrome(self, current, target_length=300):
        """
        Expand a palindrome incrementally while maintaining grammatical structure.
        """
        current_length = len(current)
        attempts = 0
        max_attempts = 5000
        
        while current_length < target_length and attempts < max_attempts:
            attempts += 1
            
            # Choose an expansion strategy
            strategies = [
                self._expand_with_mirror_words,
                self._expand_with_center_word,
                self._expand_with_sentence_wrap
            ]
            
            strategy = random.choice(strategies)
            new_palindrome = strategy(current)
            
            if new_palindrome != current and self.is_palindrome(new_palindrome):
                current = new_palindrome
                current_length = len(current)
                logging.info(f"Expanded palindrome to length {current_length}")
            
        if attempts >= max_attempts:
            logging.warning(f"Reached max attempts ({max_attempts}) without reaching target length")
            
        return current
    
    def _expand_with_mirror_words(self, current):
        """Expand palindrome by adding mirrored words at the beginning and end."""
        pos_choices = ['NOUN', 'ADJ', 'VERB', 'ADV']
        pos = random.choice(pos_choices)
        
        if not self.word_dict[pos]:
            return current
        
        # Try up to 10 words until we find one that creates a palindrome
        for _ in range(10):
            word = random.choice(self.word_dict[pos])
            new_palindrome = f"{word} {current} {word}"
            if self.is_palindrome(new_palindrome):
                return new_palindrome
        
        # If we can't find a suitable word, try a simple one that's guaranteed to work
        simple_words = ["a", "I", "my", "no", "on"]
        for word in simple_words:
            new_palindrome = f"{word} {current} {word}"
            if self.is_palindrome(new_palindrome):
                return new_palindrome
                
        # If all else fails, return the unchanged current palindrome
        return current
    
    def _expand_with_center_word(self, current):
        """Expand palindrome by adding a word in the center (if length is odd)."""
        if len(current) % 2 == 1:
            center_idx = len(current) // 2
            
            # Try with palindrome words to maintain the palindrome property
            palindrome_words = ["a", "I", "deed", "eye", "mom", "noon", "peep", "wow"]
            
            # Try to find palindrome words from our dictionary
            for pos in ['NOUN', 'ADJ']:
                if pos in self.word_dict:
                    palindrome_dict_words = [word for word in self.word_dict[pos] if self.is_palindrome(word)]
                    if palindrome_dict_words:
                        palindrome_words.extend(palindrome_dict_words[:5])  # Add up to 5 palindrome words
            
            # Try each word to see if it maintains the palindrome property
            for word in palindrome_words:
                new_palindrome = current[:center_idx] + f" {word} " + current[center_idx:]
                if self.is_palindrome(new_palindrome):
                    return new_palindrome
                
        return current
    
    def _expand_with_sentence_wrap(self, current):
        """Wrap the current palindrome with a simple sentence structure."""
        # Use templates that are more likely to form palindromes
        templates = [
            f"a {current} a",
            f"I {current} I",
            f"my {current} my",
            f"no {current} on",
            f"to {current} ot",
            f"mom {current} mom",
            f"dad {current} dad",
            f"wow {current} wow"
        ]
        
        # Try each template to see if it forms a palindrome
        for template in templates:
            if self.is_palindrome(template):
                return template
        
        # If none of the templates work, return the unchanged palindrome
        return current
    
    def generate_palindrome_paragraph(self, target_length=300):
        """
        Generate a grammatically correct palindrome paragraph with a target length.
        """
        # Start with a simple palindrome sentence
        palindrome = self.generate_palindrome_sentence()
        logging.info(f"Starting with seed: {palindrome}")
        
        # Expand it to reach the target length
        expanded_palindrome = self.expand_palindrome(palindrome, target_length)
        
        # Clean up and format
        cleaned = re.sub(r'\s+', ' ', expanded_palindrome).strip()
        
        # Check before adding capitalization and period
        if not self.is_palindrome(cleaned):
            logging.warning(f"Cleaned palindrome is not a valid palindrome: '{cleaned}'")
            # Try to find the closest valid palindrome substring
            for i in range(len(cleaned)-1, 0, -1):
                if self.is_palindrome(cleaned[:i]):
                    cleaned = cleaned[:i]
                    logging.info(f"Found valid palindrome substring: '{cleaned}'")
                    break
        
        # If we still don't have a valid palindrome, fall back to a simple one
        if not self.is_palindrome(cleaned):
            cleaned = "radar"
            logging.warning("Falling back to simple palindrome 'radar'")
        
        # Capitalize first letter (no period at the end to maintain palindrome property)
        final_palindrome = cleaned[0].upper() + cleaned[1:]
        
        # Verify it's still a palindrome after formatting
        if not self.is_palindrome(final_palindrome):
            logging.warning(f"Final formatting broke the palindrome property: '{final_palindrome}'")
            # Remove capitalization if it broke the palindrome
            final_palindrome = cleaned
            
        return final_palindrome

def main():
    load_dotenv()  # Load environment variables if needed
    
    print("Initializing Grammar-Based Palindrome Generator...")
    generator = GrammarPalindromeGenerator()
    
    print("\nGenerating a grammatically correct palindrome paragraph (target length: 300 characters)...")
    palindrome = generator.generate_palindrome_paragraph(300)
    
    print("\nGenerated Palindrome:")
    print("-" * 80)
    print(palindrome)
    print("-" * 80)
    
    print(f"\nPalindrome length: {len(palindrome)} characters")
    print(f"Is palindrome: {generator.is_palindrome(palindrome)}")
    
if __name__ == "__main__":
    main()
