#!/usr/bin/env python3
"""
Final test script for the palindrome generator.
This script demonstrates the full capabilities of the improved grammar-based
palindrome generator with all enhancements.
"""

import time
import random
from grammar_palindrome_generator import GrammarPalindromeGenerator
from grammar_validator import GrammarValidator

def test_palindrome_generation(target_length=150, attempts=3):
    """Test generating palindromes with target length."""
    generator = GrammarPalindromeGenerator()
    validator = GrammarValidator()
    
    print(f"\n=== Testing Palindrome Generation (Target Length: {target_length}) ===\n")
    
    best_palindrome = None
    best_length = 0
    best_score = 0
    best_time = float('inf')
    
    for i in range(attempts):
        print(f"Attempt {i+1}:")
        
        start_time = time.time()
        palindrome = generator.generate_palindrome_paragraph(target_length)
        generation_time = time.time() - start_time
        
        # Validate the palindrome
        is_palindrome = generator.is_palindrome(palindrome)
        grammar_score = validator.score_grammatical_quality(palindrome)
        
        print(f"  Result: \"{palindrome}\"")
        print(f"  Length: {len(palindrome)} characters")
        print(f"  Is palindrome: {is_palindrome}")
        print(f"  Grammar score: {grammar_score}/100")
        print(f"  Generation time: {generation_time:.2f} seconds\n")
        
        # Keep track of the best palindrome
        if len(palindrome) > best_length or (len(palindrome) == best_length and grammar_score > best_score):
            best_palindrome = palindrome
            best_length = len(palindrome)
            best_score = grammar_score
            best_time = generation_time
    
    if attempts > 1 and best_palindrome:
        print("=== Best Palindrome ===")
        print(f"\"{best_palindrome}\"")
        print(f"Length: {best_length} characters")
        print(f"Grammar score: {best_score}/100")
        print(f"Generation time: {best_time:.2f} seconds")

def test_grammar_improvement():
    """Test the grammar improvement capabilities."""
    print("\n=== Testing Grammar Improvement ===\n")
    
    validator = GrammarValidator()
    
    # Test palindromes including some with poor grammar
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
        print(f"Original: \"{palindrome}\"")
        print(f"Grammar score: {score}/100")
        
        improved, improved_score = validator.improve_palindrome_grammar(palindrome)
        
        if improved != palindrome:
            print(f"Improved: \"{improved}\"")
            print(f"Improved score: {improved_score}/100")
            print(f"Is still palindrome: {validator._is_palindrome(improved)}")
        else:
            print("No improvement could be made while maintaining palindrome property.")
            
        print()

def main():
    """Run all tests."""
    print("======= PALINDROME SENTENCE GENERATOR FINAL TEST =======")
    
    # Test grammar improvement
    test_grammar_improvement()
    
    # Test palindrome generation with different lengths
    test_palindrome_generation(target_length=50, attempts=1)
    test_palindrome_generation(target_length=100, attempts=1)
    test_palindrome_generation(target_length=200, attempts=2)
    
if __name__ == "__main__":
    main()
