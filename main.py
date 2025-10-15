#!/usr/bin/env python3
"""
Palindrome Sentence Generator - Main Entry Point

This script generates character-level palindromes that read the same forwards
and backwards when all spaces, punctuation, and capitalization are removed.

The generator uses a constructive approach that guarantees valid palindromes
every time, unlike LLM-based approaches which have a near-zero success rate.
"""

import argparse
import sys
from palindrome_generator import generate_palindrome
from validator import is_full_palindrome, normalize


def show_palindrome_info(palindrome, normalized):
    """Display information about a palindrome."""
    print(f"\n{'='*60}")
    print(f"Generated Palindrome:")
    print(f"{'='*60}")
    print(f"\n{palindrome}\n")
    print(f"{'='*60}")
    print(f"Normalized length: {len(normalized)} characters")
    
    # Visual representation of the palindrome pattern
    if len(normalized) <= 100:
        print(f"\nNormalized text: {normalized}")
        print("\nPalindrome visualization:")
        middle = len(normalized) // 2
        visual = normalized[:middle] + "|" + normalized[middle:]
        print(visual)
        print(" " * middle + "^")
        print(f"{'Center point':^{len(normalized)}}")
    else:
        # For long palindromes, show first and last parts
        print(f"\nNormalized (showing first/last 50 chars):")
        print(f"Start: {normalized[:50]}")
        print(f"End:   {normalized[-50:]}")
        print(f"\nVerifying middle matches...")
        middle = len(normalized) // 2
        print(f"Middle segment: {normalized[middle-10:middle+10]}")

def main():
    """Main entry point for the palindrome generator."""
    parser = argparse.ArgumentParser(
        description="Generate character-level palindromic text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Generate default 60-char palindrome
  %(prog)s --length 100             # Generate 100-char palindrome
  %(prog)s --method sentence        # Use sentence-style generation
  %(prog)s --count 5                # Generate 5 palindromes
  %(prog)s --seed 42                # Use seed for reproducibility
        """
    )
    
    parser.add_argument(
        "--length", 
        type=int, 
        default=60,
        help="Minimum length of the normalized palindrome (default: 60)"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=['auto', 'core', 'sentence', 'mirror'],
        default='auto',
        help="Generation method: 'core' (build from palindromic word), "
             "'sentence' (use known phrases), 'mirror' (mirror a phrase), "
             "'auto' (random choice) (default: auto)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of palindromes to generate (default: 1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (optional)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed information"
    )
    parser.add_argument(
        "--readable",
        action="store_true",
        help="Produce more readable output by adding symmetric spacing and punctuation"
    )
    
    args = parser.parse_args()
    
    print("Palindrome Sentence Generator")
    print("="*60)
    print(f"Configuration:")
    print(f"  Minimum length: {args.length} characters")
    print(f"  Method: {args.method}")
    print(f"  Count: {args.count}")
    if args.seed is not None:
        print(f"  Random seed: {args.seed}")
    print()
    
    # Track success rate
    successful = 0
    
    for i in range(args.count):
        if args.count > 1:
            print(f"\n{'#'*60}")
            print(f"Generating palindrome {i+1}/{args.count}...")
            print(f"{'#'*60}")
        
        try:
            # Generate palindrome
            palindrome, normalized, is_valid = generate_palindrome(
                min_length=args.length,
                method=args.method,
                seed=args.seed,
                use_deterministic_spacing=args.readable
            )
            
            if is_valid:
                successful += 1
                print("✅ Valid palindrome generated!")
            else:
                print("❌ Generated text is not a valid palindrome!")
                if args.verbose:
                    print(f"   This should never happen with constructive generation.")
            
            show_palindrome_info(palindrome, normalized)
            
            # Verify with validator
            if args.verbose:
                verified = is_full_palindrome(palindrome)
                print(f"\nDouble-check validation: {'✅ PASS' if verified else '❌ FAIL'}")
                if not verified:
                    print(f"Expected: {normalized}")
                    print(f"Reversed: {normalized[::-1]}")
            
        except Exception as e:
            print(f"❌ Error generating palindrome: {str(e)}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    # Summary
    if args.count > 1:
        print(f"\n{'='*60}")
        print(f"Summary: {successful}/{args.count} palindromes generated successfully")
        print(f"Success rate: {100*successful/args.count:.1f}%")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
