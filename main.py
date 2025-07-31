#!/usr/bin/env python3
"""
Palindrome Paragraph Generator - Main Script

This script provides a command-line interface for generating palindrome paragraphs
using either the basic palindrome generator or the grammar-based palindrome generator.
"""

import argparse
import logging
import time
from palindrome_generator import PalindromeGenerator
from grammar_palindrome_generator import GrammarPalindromeGenerator

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        filename='palindrome.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Generate palindrome paragraphs.')
    
    parser.add_argument(
        '--length', 
        type=int, 
        default=300,
        help='Target length of the palindrome paragraph (default: 300 characters).'
    )
    
    parser.add_argument(
        '--method', 
        type=str,
        choices=['basic', 'grammar'],
        default='grammar',
        help='The method to use for palindrome generation (default: grammar).'
    )
    
    parser.add_argument(
        '--output', 
        type=str,
        help='Output file to save the generated palindrome.'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose output.'
    )
    
    return parser.parse_args()

def main():
    """Main function to run the palindrome generator."""
    setup_logging()
    args = parse_arguments()
    
    if args.verbose:
        print(f"Generating palindrome using '{args.method}' method with target length {args.length}...")
    
    start_time = time.time()
    
    # Choose the generator based on the method
    if args.method == 'basic':
        generator = PalindromeGenerator()
        palindrome = generator.generate_grammatical_palindrome(args.length)
    else:  # args.method == 'grammar'
        generator = GrammarPalindromeGenerator()
        palindrome = generator.generate_palindrome_paragraph(args.length)
    
    end_time = time.time()
    generation_time = end_time - start_time
    
    # Display the generated palindrome
    print("\nGenerated Palindrome:")
    print("-" * 80)
    print(palindrome)
    print("-" * 80)
    
    # Display statistics
    print(f"\nPalindrome length: {len(palindrome)} characters")
    print(f"Is palindrome: {generator.is_palindrome(palindrome)}")
    
    if args.verbose:
        print(f"Generation time: {generation_time:.2f} seconds")
    
    # Save to output file if specified
    if args.output:
        try:
            with open(args.output, 'w') as f:
                f.write(palindrome)
            print(f"Palindrome saved to '{args.output}'")
        except Exception as e:
            print(f"Error saving to output file: {e}")
    
if __name__ == "__main__":
    main()
