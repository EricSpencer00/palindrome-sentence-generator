#!/usr/bin/env python3
"""
Simple Palindrome Test

This script tests a simple palindrome generation to verify functionality.
"""

def is_palindrome(text):
    """Check if a string is a palindrome (ignoring spaces and punctuation)."""
    text = ''.join(c.lower() for c in text if c.isalnum())
    return text == text[::-1]

def main():
    # Test palindrome checking
    test_cases = [
        "A man, a plan, a canal: Panama",
        "Race a car",
        "No lemon, no melon",
        "Was it a car or a cat I saw?",
        "Never odd or even"
    ]
    
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
