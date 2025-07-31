#!/usr/bin/env python3
"""
Simple Palindrome Test

This script tests the palindrome generators to verify functionality.
"""

import sys
import argparse
from palindrome_generator import PalindromeGenerator
from grammar_palindrome_generator import GrammarPalindromeGenerator

def is_palindrome(text):
    """Check if a string is a palindrome (ignoring spaces and punctuation)."""
    text = ''.join(c.lower() for c in text if c.isalnum())
    return text == text[::-1]

def test_is_palindrome():
    """Test the palindrome checking function."""
    # Test palindrome checking
    test_cases = [
        ("A man, a plan, a canal: Panama", True),
        ("Race a car", False),
        ("No lemon, no melon", True),
        ("Was it a car or a cat I saw?", True),
        ("Never odd or even", True),
        ("hello world", False),
        ("radar", True),
        ("a", True),
        ("", True)  # Empty string is technically a palindrome
    ]
    
    failures = 0
    for text, expected in test_cases:
        result = is_palindrome(text)
        if result != expected:
            print(f"FAILED: '{text}' - Expected: {expected}, Got: {result}")
            failures += 1
        else:
            print(f"PASSED: '{text}'")
    
    print(f"\nPalindrome check tests: {len(test_cases) - failures}/{len(test_cases)} passed")
    return failures == 0

def test_generators(target_length=50):
    """Test both palindrome generators."""
    print("\nTesting Basic Palindrome Generator...")
    basic_gen = PalindromeGenerator()
    basic_palindrome = basic_gen.generate_grammatical_palindrome(target_length)
    basic_result = is_palindrome(basic_palindrome)
    basic_length = len(basic_palindrome)
    
    print(f"Generated: '{basic_palindrome}'")
    print(f"Length: {basic_length} characters")
    print(f"Is palindrome: {basic_result}")
    
    print("\nTesting Grammar-Based Palindrome Generator...")
    grammar_gen = GrammarPalindromeGenerator()
    grammar_palindrome = grammar_gen.generate_palindrome_paragraph(target_length)
    grammar_result = is_palindrome(grammar_palindrome)
    grammar_length = len(grammar_palindrome)
    
    print(f"Generated: '{grammar_palindrome}'")
    print(f"Length: {grammar_length} characters")
    print(f"Is palindrome: {grammar_result}")
    
    success = basic_result and grammar_result
    print(f"\nGenerator tests: {'PASSED' if success else 'FAILED'}")
    return success

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Test palindrome generators.')
    
    parser.add_argument(
        '--length', 
        type=int, 
        default=50,
        help='Target length for generated palindromes (default: 50 characters).'
    )
    
    parser.add_argument(
        '--check-only', 
        action='store_true',
        help='Only test the is_palindrome function without generating palindromes.'
    )
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    print("===== Palindrome Test Suite =====\n")
    
    # Always test the is_palindrome function
    check_success = test_is_palindrome()
    
    # Test generators if requested
    if not args.check_only:
        gen_success = test_generators(args.length)
        success = check_success and gen_success
    else:
        success = check_success
    
    # Return exit code based on test success
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
    
    print("Testing palindrome detection:")
    for test in test_cases:
        print(f"'{test}' is palindrome: {is_palindrome(test)}")
    
    # Create a simple palindrome
    words = ["radar", "level", "refer", "civic", "madam"]
    
    for word in words:
        print(f"\nBuilding palindrome with seed: {word}")
        # Create a simple expansion
        expansion = f"I {word} I"
        print(f"Expansion: '{expansion}'")
        print(f"Is palindrome: {is_palindrome(expansion)}")
        
        # Create another expansion
        expansion2 = f"My {word} my"
        print(f"Expansion: '{expansion2}'")
        print(f"Is palindrome: {is_palindrome(expansion2)}")

if __name__ == "__main__":
    main()
