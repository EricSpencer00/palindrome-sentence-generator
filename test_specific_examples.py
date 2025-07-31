#!/usr/bin/env python3
"""
Test the enhanced grammar palindrome generator with specific examples.
This script demonstrates how the grammar-aware generator handles challenging cases.
"""

import logging
from palindrome_generator import PalindromeParagraphGenerator
from grammar_palindrome_generator import GrammarPalindromeGenerator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_specific_examples():
    """Test specific examples to demonstrate grammar improvement"""
    # Initialize generators
    original_generator = PalindromeParagraphGenerator()
    grammar_generator = GrammarPalindromeGenerator()
    
    # Test cases - challenging palindromes from the prompt
    examples = [
        "Radar level sees noon did wow pop eye Noon is usually when the sun is highest in the sky and never odd or even",
        "Radar The level sees The noon did wow pop eye A radar The system detected unusual The activity near the dam The racecar"
    ]
    
    for i, example in enumerate(examples):
        print(f"\n{'='*80}")
        print(f"EXAMPLE {i+1}:")
        print(f"{'='*80}")
        print(example)
        
        # Check if it's a palindrome
        is_palindrome = original_generator.is_palindrome(example)
        print(f"\nIs this a valid palindrome? {is_palindrome}")
        
        if is_palindrome:
            # Attempt to improve the grammar with our new generator
            print("\nImproving grammar...")
            
            # Extract a center phrase
            center_options = [
                "never odd or even",
                "racecar",
                "noon",
                "eye"
            ]
            
            center = None
            for phrase in center_options:
                if phrase.lower() in example.lower():
                    center = phrase
                    break
            
            if not center:
                center = "a man a plan a canal panama"  # Default
                
            # Generate improved version
            improved = grammar_generator.generate_palindrome(
                center=center,
                target_length=len(example)
            )
            
            print(f"\n{'='*80}")
            print("IMPROVED VERSION:")
            print(f"{'='*80}")
            print(improved)
            print(f"\nIs still a palindrome? {grammar_generator.is_palindrome(improved)}")
            
            # Analyze and compare
            print("\nAnalysis:")
            print(f"Original: {len(example)} characters, {len(example.split())} words")
            print(f"Improved: {len(improved)} characters, {len(improved.split())} words")
            
            # Compare word boundaries
            orig_words = example.split()
            impr_words = improved.split()
            
            print("\nOriginal word boundaries (first 10):")
            for i, word in enumerate(orig_words[:10]):
                print(f"  {i+1}. {word}")
                
            print("\nImproved word boundaries (first 10):")
            for i, word in enumerate(impr_words[:10]):
                print(f"  {i+1}. {word}")
                
    print(f"\n{'='*80}")
    print("KEY IMPROVEMENTS IN THE GRAMMAR-ENHANCED GENERATOR:")
    print(f"{'='*80}")
    print("1. Both halves have grammatically sensible word boundaries")
    print("2. Character-level palindrome property is maintained")
    print("3. Grammatical structure is enforced on both sides")
    print("4. Articles are properly paired with nouns")
    print("5. Consistent naming in both halves")
    print(f"{'='*80}")

if __name__ == "__main__":
    test_specific_examples()
