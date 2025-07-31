#!/usr/bin/env python3
"""
Test script for palindrome generator improvements.
"""

import time
from grammar_palindrome_generator import GrammarPalindromeGenerator
from grammar_validator import GrammarValidator

def test_seed_based_generation():
    """Test generating palindromes with known good seeds."""
    generator = GrammarPalindromeGenerator()
    
    # Use a known good seed
    current = "A man a plan a canal Panama"
    print(f"Starting with seed: '{current}'")
    print(f"Is palindrome: {generator.is_palindrome(current)}")
    
    # Try to expand it
    for i in range(5):
        new_current = generator._expand_with_mirror_words(current)
        if new_current != current:
            print(f"Expansion {i+1}: '{new_current}'")
            print(f"Is palindrome: {generator.is_palindrome(new_current)}")
            current = new_current
        else:
            print(f"No expansion at attempt {i+1}")
    
    print("\n" + "-" * 80 + "\n")
    
    # Try other expansion strategies
    current = "Step on no pets"
    print(f"Starting with seed: '{current}'")
    print(f"Is palindrome: {generator.is_palindrome(current)}")
    
    # Try sentence wrap
    wrapped = generator._expand_with_sentence_wrap(current)
    print(f"Sentence wrap: '{wrapped}'")
    print(f"Is palindrome: {generator.is_palindrome(wrapped)}")
    
    # Try center word
    centered = generator._expand_with_center_word(current)
    print(f"Center word: '{centered}'")
    print(f"Is palindrome: {generator.is_palindrome(centered)}")

def test_grammar_improvements():
    """Test grammar validator improvements."""
    validator = GrammarValidator()
    
    test_palindromes = [
        "A man a plan a canal Panama",
        "Was it a car or a cat I saw",
        "Madam I'm Adam",
        "a a a a a a a a a radar a a a a a a a a a",
        "Mom wow mom eye sees mom wow mom",
        "Step on no pets kayak step on no pets"
    ]
    
    for palindrome in test_palindromes:
        score = validator.score_grammatical_quality(palindrome)
        print(f"\nOriginal: \"{palindrome}\"")
        print(f"Grammar score: {score}/100")
        
        improved, improved_score = validator.improve_palindrome_grammar(palindrome)
        
        if improved != palindrome:
            print(f"Improved: \"{improved}\"")
            print(f"Improved score: {improved_score}/100")
            print(f"Is still palindrome: {validator._is_palindrome(improved)}")
        else:
            print("No improvement could be made while maintaining palindrome property.")

def test_full_generation():
    """Test generating multiple full palindromes."""
    generator = GrammarPalindromeGenerator()
    
    for length in [50, 100, 150]:
        print(f"\nGenerating palindrome with target length {length}:")
        start_time = time.time()
        palindrome = generator.generate_palindrome_paragraph(length)
        generation_time = time.time() - start_time
        
        print(f"Result: \"{palindrome}\"")
        print(f"Length: {len(palindrome)} characters")
        print(f"Is palindrome: {generator.is_palindrome(palindrome)}")
        print(f"Generation time: {generation_time:.2f} seconds")
        
        # Check grammar quality
        validator = GrammarValidator()
        score = validator.score_grammatical_quality(palindrome)
        print(f"Grammar score: {score}/100")

if __name__ == "__main__":
    print("=== Testing Seed-Based Generation ===")
    test_seed_based_generation()
    
    print("=== Testing Grammar Improvements ===")
    test_grammar_improvements()
    
    print("\n=== Testing Full Generation ===")
    test_full_generation()
