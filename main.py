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
from grammar_validator import GrammarValidator

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
    
    parser.add_argument(
        '--improve-grammar',
        action='store_true',
        help='Attempt to improve grammar of the generated palindrome.'
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
        
        # Make multiple attempts if needed to reach target length
        best_palindrome = None
        best_length = 0
        
        for attempt in range(3):
            if args.verbose:
                print(f"Generation attempt {attempt+1}...")
                
            current_palindrome = generator.generate_palindrome_paragraph(args.length)
            current_length = len(current_palindrome)
            
            if current_length > best_length:
                best_palindrome = current_palindrome
                best_length = current_length
                
            # If we've reached at least 80% of target length, stop
            if current_length >= args.length * 0.8:
                break
                
        palindrome = best_palindrome
    
    end_time = time.time()
    generation_time = end_time - start_time
    
    # Apply grammar improvements if requested
    if args.improve_grammar:
        if args.verbose:
            print("Attempting to improve grammatical structure...")
        
        grammar_start_time = time.time()
        validator = GrammarValidator()
        initial_score = validator.score_grammatical_quality(palindrome)
        
        # Try multiple improvement attempts
        best_score = initial_score
        best_palindrome = palindrome
        
        for attempt in range(3):  # Try up to 3 improvement attempts
            improved_palindrome, improved_score = validator.improve_palindrome_grammar(palindrome)
            
            if improved_score > best_score and generator.is_palindrome(improved_palindrome):
                best_palindrome = improved_palindrome
                best_score = improved_score
                
                if args.verbose:
                    print(f"Grammar improved (attempt {attempt+1}): {initial_score} → {improved_score}/100")
            
            # Use the improved palindrome for the next improvement attempt
            palindrome = improved_palindrome
                
        # Use the best palindrome found
        if best_score > initial_score:
            palindrome = best_palindrome
        elif args.verbose:
            print("No grammar improvements could be made while maintaining palindrome property.")
        
        grammar_time = time.time() - grammar_start_time
        if args.verbose:
            print(f"Grammar analysis time: {grammar_time:.2f} seconds")
    
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
        
        # If we have a grammar validator, show grammar score
        if 'validator' in locals():
            grammar_score = validator.score_grammatical_quality(palindrome)
            print(f"Grammar score: {grammar_score}/100")
            
            # Show grammar suggestions
            suggestions = validator.suggest_grammar_improvements(palindrome)
            if suggestions:
                print("\nGrammar suggestions:")
                for suggestion in suggestions[:3]:  # Limit to top 3 suggestions
                    print(f"- {suggestion}")
    
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
