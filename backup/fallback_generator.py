#!/usr/bin/env python3
# filepath: /Users/ericspencer/GitHub/palindrome-sentence-generator/fallback_generator.py
"""
Fallback Palindrome Generator

This module provides a local fallback generator that uses NLTK to create palindromes
when the LLM-based generation fails or is unavailable.
"""

import random
import re
import nltk
from nltk.corpus import words, wordnet
from validator import normalize

# Ensure NLTK resources are available
try:
    nltk.data.find('corpora/words')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('words')
    nltk.download('wordnet')

# Common palindromes to use as building blocks
PALINDROME_WORDS = [
    "civic", "level", "radar", "refer", "madam", "deed", "eye", "mom", "dad", 
    "noon", "tenet", "stats", "kayak", "rotator", "wow", "peep", "sees", "racecar"
]

# Common words to use in palindrome sentences
COMMON_WORDS = {
    'nouns': ['eye', 'dad', 'mom', 'noon', 'madam', 'sir', 'sis', 'eve', 'bob', 'anna', 'otto', 'level', 'deed', 'noon', 'rotator'],
    'verbs': ['sees', 'did', 'nod', 'peek', 'won', 'draw', 'refer', 'stare', 'live'],
    'adjectives': ['mad', 'raw', 'civic', 'evil', 'rare', 'red', 'lol'],
    'pronouns': ['I', 'me', 'no', 'on', 'we'],
    'articles': ['a'],
    'prepositions': ['at', 'no', 'to', 'in', 'by'],
}

def is_palindrome(text):
    """Check if a text is a palindrome when normalized."""
    normalized = normalize(text)
    return normalized == normalized[::-1]

def get_word_bank():
    """Create a bank of words that could be useful for palindromes."""
    word_bank = {}
    
    # Add common palindrome words and predefined words
    for pos, wordlist in COMMON_WORDS.items():
        word_bank[pos] = set(wordlist)
    
    # Add some short words from NLTK that are likely to be useful
    try:
        english_words = [w.lower() for w in words.words() if len(w) <= 6 and w.isalpha()]
        # Add a subset of the words to avoid memory issues
        sample_size = min(5000, len(english_words))
        sampled_words = random.sample(english_words, sample_size)
        
        # Simple categorization based on word endings and common patterns
        for word in sampled_words:
            if word.endswith(('er', 'or', 'ist', 'ian')):
                word_bank['nouns'].add(word)
            elif word.endswith(('ed', 'ing')):
                word_bank['verbs'].add(word)
            elif word.endswith(('ly', 'ous', 'ful', 'ive')):
                word_bank['adjectives'].add(word)
            elif len(word) <= 3:  # Short words often used as prepositions or articles
                word_bank['prepositions'].add(word)
    except Exception as e:
        print(f"Warning: Error processing NLTK words: {e}")
    
    return word_bank

def reverse_word(word):
    """Return the reversed version of a word."""
    return word[::-1]

def build_palindrome_from_center(center, min_length=50):
    """Build a palindrome starting from a center word or phrase."""
    word_bank = get_word_bank()
    
    # Start with the center
    palindrome = center
    normalized = normalize(palindrome)
    
    # Keep adding words until we reach the minimum length
    while len(normalized) < min_length:
        # Randomly choose a part of speech
        pos = random.choice(list(word_bank.keys()))
        
        # Get a random word of that type
        if word_bank[pos]:
            word = random.choice(list(word_bank[pos]))
            reversed_word = reverse_word(word)
            
            # Add the word to the beginning and its reverse to the end
            palindrome = word + " " + palindrome + " " + reversed_word
            normalized = normalize(palindrome)
        else:
            # If no words of this type, try another type
            continue
    
    return palindrome

def generate_palindrome_pairs(min_length=50):
    """Generate palindromes using word pairs."""
    # Start with a center word
    center = random.choice(PALINDROME_WORDS)
    palindrome = center
    normalized = normalize(palindrome)
    
    # Keep adding word pairs until we reach the minimum length
    while len(normalized) < min_length:
        # Choose a random letter
        letter = chr(random.randint(97, 122))  # a-z
        
        # Create mirrored structure
        if random.random() < 0.3:
            # Add "word" structure
            palindrome = letter + palindrome + letter
        else:
            # Add a more complex structure
            structure = random.choice(["no", "on", "to", "at", "in", "is", "as"])
            palindrome = structure + " " + palindrome + " " + structure[::-1]
        
        normalized = normalize(palindrome)
    
    return palindrome

def generate_local_palindrome(min_length=50, attempts=5):
    """Generate a palindrome locally using NLTK resources."""
    best_palindrome = None
    
    for i in range(attempts):
        try:
            # Alternate between different strategies
            if i % 2 == 0:
                # Choose a random palindrome word as the center
                center = random.choice(PALINDROME_WORDS)
                palindrome_text = build_palindrome_from_center(center, min_length)
            else:
                palindrome_text = generate_palindrome_pairs(min_length)
            
            # Verify it's a palindrome
            if is_palindrome(palindrome_text):
                # If no best palindrome yet or this one is better (closer to target length)
                if (not best_palindrome or 
                    abs(len(normalize(palindrome_text)) - min_length) < 
                    abs(len(normalize(best_palindrome)) - min_length)):
                    best_palindrome = palindrome_text
        except Exception as e:
            print(f"Error in generation attempt {i+1}: {e}")
            continue
    
    if best_palindrome:
        # Add some punctuation and capitalization to make it more readable
        words = best_palindrome.split()
        sentence = ""
        current_sentence = ""
        
        for i, word in enumerate(words):
            current_sentence += word + " "
            
            # End a sentence randomly or at the end
            if i == len(words) - 1 or (i > 3 and random.random() < 0.2):
                if current_sentence:
                    # Capitalize first letter
                    current_sentence = current_sentence[0].upper() + current_sentence[1:]
                    # Add period
                    current_sentence = current_sentence.strip() + ". "
                    sentence += current_sentence
                    current_sentence = ""
        
        return sentence.strip()
    
    # If all attempts failed, return a simple palindrome
    return "Eva, can I see bees in a cave? No evil, I live on. Eva, can I see bees in a cave?"

# Explicitly export the function for use in main.py
__all__ = ['generate_local_palindrome']
