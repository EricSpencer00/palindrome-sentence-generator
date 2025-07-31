#!/usr/bin/env python3
"""
Compare original palindrome generator with the enhanced grammar version.
This script demonstrates the improvements in grammar while maintaining
the palindrome property.
"""

import logging
import time
from palindrome_generator import PalindromeParagraphGenerator
from grammar_palindrome_generator import GrammarPalindromeGenerator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compare_generators():
    """Compare original and enhanced grammar-aware generators"""
    print("\n" + "="*80)
    print("PALINDROME GENERATOR COMPARISON")
    print("="*80)
    
    # Initialize generators
    original_generator = PalindromeParagraphGenerator()
    grammar_generator = GrammarPalindromeGenerator()
    
    # Common center phrase for comparison
    center_phrase = "a man a plan a canal panama"
    target_length = 200
    
    # Original generator - bidirectional method
    print("\nGenerating palindrome with original bidirectional method...")
    start_time = time.time()
    original_palindrome = original_generator.generate_with_llm(
        middle=center_phrase,
        target_length=target_length,
        max_attempts=3
    )
    original_time = time.time() - start_time
    
    # Enhanced grammar generator
    print("\nGenerating palindrome with enhanced grammar-aware method...")
    start_time = time.time()
    grammar_palindrome = grammar_generator.generate_palindrome(
        center=center_phrase,
        target_length=target_length
    )
    grammar_time = time.time() - start_time
    
    # Display results
    print("\n" + "="*80)
    print("ORIGINAL PALINDROME:")
    print("="*80)
    print(original_palindrome)
    print(f"\nTime taken: {original_time:.2f} seconds")
    print(f"Length: {len(original_palindrome)} characters")
    print(f"Word count: {len(original_palindrome.split())} words")
    print(f"Is palindrome: {original_generator.is_palindrome(original_palindrome)}")
    
    print("\n" + "="*80)
    print("GRAMMAR-ENHANCED PALINDROME:")
    print("="*80)
    print(grammar_palindrome)
    print(f"\nTime taken: {grammar_time:.2f} seconds")
    print(f"Length: {len(grammar_palindrome)} characters")
    print(f"Word count: {len(grammar_palindrome.split())} words")
    print(f"Is palindrome: {grammar_generator.is_palindrome(grammar_palindrome)}")
    
    # Analyze grammar structure
    print("\n" + "="*80)
    print("GRAMMAR ANALYSIS:")
    print("="*80)
    
    # Simple analysis - split in half and check word structure
    def analyze_halves(text, generator):
        words = text.split()
        mid_point = len(words) // 2
        first_half = " ".join(words[:mid_point])
        second_half = " ".join(words[mid_point:])
        
        print(f"First half:  {first_half}")
        print(f"Second half: {second_half}")
        
        # Clean for palindrome check
        cleaned = generator.clean_text(text)
        mid_char = len(cleaned) // 2
        first_chars = cleaned[:mid_char]
        second_chars = cleaned[mid_char:]
        
        # Check character-level palindrome property
        print(f"Character-level palindrome check: {first_chars == second_chars[::-1]}")
    
    print("\nOriginal palindrome structure:")
    analyze_halves(original_palindrome, original_generator)
    
    print("\nGrammar-enhanced palindrome structure:")
    analyze_halves(grammar_palindrome, grammar_generator)
    
    print("\n" + "="*80)

if __name__ == "__main__":
    compare_generators()
