#!/usr/bin/env python3
"""
Fallback Palindrome Generator

This script provides a fallback for generating palindromes when the API-based
approach fails. It uses the existing grammar palindrome generator.
"""

import argparse
import logging
import time
from grammar_palindrome_generator import GrammarPalindromeGenerator
from validator import is_full_palindrome, normalize

def generate_local_palindrome(target_length=150, max_attempts=20):
    """Generate a palindrome using the local grammar-based generator."""
    generator = GrammarPalindromeGenerator()
    
    print(f"Attempting to generate a palindrome with target length {target_length}...")
    
    best_palindrome = None
    best_length = 0
    
    for i in range(max_attempts):
        print(f"Local generation attempt {i+1}...")
        try:
            current = generator.generate_palindrome_paragraph(target_length)
            
            # Check if it's a valid palindrome
            if is_full_palindrome(current):
                normalized = normalize(current)
                current_length = len(normalized)
                
                print(f"Generated valid palindrome with length {current_length}")
                
                if current_length > best_length:
                    best_palindrome = current
                    best_length = current_length
                    
                    # If we've reached at least 80% of target length, we can stop
                    if current_length >= target_length * 0.8:
                        break
            else:
                print("Generated text is not a valid palindrome.")
                
        except Exception as e:
            print(f"Error during local generation: {str(e)}")
    
    if best_palindrome:
        return best_palindrome
    else:
        return "No valid palindrome generated after multiple attempts."

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate palindromes using local generator")
    parser.add_argument("--length", type=int, default=150, help="Target length of palindrome")
    parser.add_argument("--attempts", type=int, default=20, help="Maximum number of generation attempts")
    
    args = parser.parse_args()
    
    palindrome = generate_local_palindrome(args.length, args.attempts)
    
    print("\nGenerated Palindrome:")
    print(palindrome)
    
    normalized = normalize(palindrome)
    print(f"\nNormalized length: {len(normalized)} characters")
    print(f"Normalized: {normalized}")
