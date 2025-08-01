#!/usr/bin/env python3
# filepath: /Users/ericspencer/GitHub/palindrome-sentence-generator/main.py
import os
import sys
import time
import argparse
from dotenv import load_dotenv
from utils import generate_palindromic_paragraph
from validator import is_full_palindrome, normalize
from fallback_generator import generate_local_palindrome
from constructive_generator import generate_constructive_palindrome

# Set this to True after we've created the fallback_generator.py
FALLBACK_AVAILABLE = True
# Set this to True now that we've created the constructive_generator.py
CONSTRUCTIVE_AVAILABLE = True

def show_palindrome_info(palindrome, normalized):
    """Display information about a palindrome."""
    print(f"\nGenerated text: {palindrome}")
    print(f"Normalized length: {len(normalized)} characters")
    print(f"Normalized text: {normalized}")
    
    # Visual representation of the palindrome pattern
    if len(normalized) <= 100:
        print("\nPalindrome visualization:")
        middle = len(normalized) // 2
        visual = normalized[:middle] + "|" + normalized[middle:]
        print(visual)
        print(" " * middle + "^" + " " * middle)
        print(" " * middle + "|" + " " * middle)
        print("Center point")

def attempt_palindrome_repair(text, min_length=60):
    """Try to repair a near-palindrome to make it valid"""
    normalized = normalize(text)
    
    # If it's already a palindrome, return it
    if normalized == normalized[::-1]:
        return text
    
    # Strategy 1: Take the first half and mirror it to create a perfect palindrome
    half_length = len(normalized) // 2
    first_half = normalized[:half_length]
    mirrored = first_half + first_half[::-1]
    
    # If the string length should be odd, add a middle character
    if len(normalized) % 2 == 1:
        middle_char = normalized[half_length]
        mirrored = first_half + middle_char + first_half[::-1]
    
    # Now try to convert this back to readable text
    # This is a challenging task as we need to insert spaces and punctuation
    
    # For now, we'll just return a simple version with spaces after every few characters
    readable = ""
    word_length = 3  # Average word length
    
    for i in range(0, len(mirrored), word_length):
        word = mirrored[i:i+word_length]
        if i == 0:
            # Capitalize the first word
            word = word.capitalize()
        readable += word + " "
        
        # Add some punctuation occasionally
        if i > 0 and i % (word_length * 3) == 0:
            readable = readable.rstrip() + ". "
    
    return readable.strip()

def enhanced_palindrome_generation(attempts=5, min_length=60, use_nltk_seed=True, model="google/gemma-3n-e2b-it:free", method="llm"):
    """Enhanced palindrome generation with multiple strategies and repair mechanisms"""
    best_palindrome = None
    best_match_percentage = 0
    best_length = 0
    
    # Choose generation method
    if method == "constructive" and CONSTRUCTIVE_AVAILABLE:
        print("\nUsing constructive generation approach...")
        # Try each strategy in the constructive generator
        strategies = ["constructive", "center_out", "mirrored", "combinatorial"]
        
        for i, strategy in enumerate(strategies):
            print(f"\nTrying {strategy} strategy (attempt {i+1})...")
            paragraph = generate_constructive_palindrome(min_length=min_length, strategy=strategy)
            
            # Validate if it's a palindrome
            normalized = normalize(paragraph)
            
            if is_full_palindrome(paragraph):
                print("\n✅ Valid full palindrome paragraph!")
                
                # Check if it meets the minimum length
                if len(normalized) >= min_length:
                    return paragraph, normalized, True
                else:
                    print(f"❌ Palindrome too short: {len(normalized)} characters (minimum: {min_length})")
                    if not best_palindrome or len(normalized) > best_length:
                        best_palindrome = paragraph
                        best_length = len(normalized)
        
        # If we tried all strategies but didn't find a good enough palindrome
        if best_palindrome:
            print("\n⚠️ Using best constructive palindrome found (didn't meet length requirement)")
            return best_palindrome, normalize(best_palindrome), False
        
        # Fall back to LLM if constructive failed
        print("\n⚠️ Constructive generation failed, trying LLM generation...")
        method = "llm"
    
    # Try LLM-based generation if specified or as fallback
    if method == "llm":
        for i in range(attempts):
            try:
                print(f"\nAttempt {i+1}...")
                
                # Generate palindrome with NLTK seed if requested
                paragraph = generate_palindromic_paragraph(
                    prompt_path="prompts/simple_palindrome.txt",
                    use_nltk_seed=use_nltk_seed,
                    min_seed_length=min_length // 2,  # Use half the minimum length for the seed
                    model=model
                )
                
                if not paragraph or paragraph.strip() == "":
                    print("Received empty response. Trying again...")
                    time.sleep(2)
                    continue
                
                # Validate if it's a palindrome
                normalized = normalize(paragraph)
                
                if is_full_palindrome(paragraph):
                    print("\n✅ Valid full palindrome paragraph!")
                    
                    # Check if it meets the minimum length
                    if len(normalized) >= min_length:
                        return paragraph, normalized, True
                    else:
                        print(f"❌ Palindrome too short: {len(normalized)} characters (minimum: {min_length})")
                        
                        # Keep track of the best (longest) palindrome we've found
                        if not best_palindrome or len(normalized) > best_length:
                            best_palindrome = paragraph
                            best_length = len(normalized)
                else:
                    # If not a palindrome, check how close it is
                    reversed_norm = normalized[::-1]
                    match_percentage = sum(a == b for a, b in zip(normalized, reversed_norm)) / len(normalized) * 100
                    print(f"❌ Not a valid palindrome. Match percentage: {match_percentage:.1f}%")
                    
                    # Store if it's the best match so far
                    if match_percentage > best_match_percentage:
                        best_palindrome = paragraph
                        best_match_percentage = match_percentage
                        best_length = len(normalized)
                        
            except Exception as e:
                print(f"Error during generation: {str(e)}")
                time.sleep(2)
    
    # Fall back to local generation if LLM fails and fallback is available
    if not best_palindrome and FALLBACK_AVAILABLE:
        print("\n⚠️ LLM generation failed, using local fallback generator...")
        try:
            paragraph = generate_local_palindrome(min_length=min_length)
            normalized = normalize(paragraph)
            if is_full_palindrome(paragraph):
                print("✅ Successfully generated a palindrome with fallback generator")
                return paragraph, normalized, True
            else:
                print("❌ Fallback generator produced an invalid palindrome")
        except Exception as e:
            print(f"Error during fallback generation: {str(e)}")
    
    # If we found any palindrome, return it
    if best_palindrome:
        normalized = normalize(best_palindrome)
        if is_full_palindrome(best_palindrome):
            print("\n⚠️ Using best palindrome found (didn't meet all requirements)")
            return best_palindrome, normalized, False
        else:
            print("\n⚠️ No valid palindrome found, attempting to repair best match...")
            repaired = attempt_palindrome_repair(best_palindrome, min_length)
            repaired_norm = normalize(repaired)
            return repaired, repaired_norm, repaired_norm == repaired_norm[::-1]
    
    # If all else fails, create a simple palindrome
    print("\n⚠️ All generation methods failed, creating a simple palindrome...")
    simple = "Madam, I'm Adam. Madam, I'm Adam."
    return simple, normalize(simple), True

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate character-level palindromic paragraphs")
    parser.add_argument("--attempts", type=int, default=5, help="Number of generation attempts")
    parser.add_argument("--min-length", type=int, default=60, help="Minimum palindrome length")
    parser.add_argument("--use-nltk-seed", action="store_true", help="Use NLTK to generate a palindrome seed")
    parser.add_argument("--model", type=str, default="google/gemma-3n-e2b-it:free", 
                        help="Model to use for generation")
    parser.add_argument("--enhanced", action="store_true", default=True, 
                        help="Use enhanced generation algorithm")
    # Add option to use the improved generator
    parser.add_argument("--use-improved", action="store_true", default=False,
                        help="Use the improved generator (enhanced multi-sentence approach)")
    # Add option to use the constructive generator
    parser.add_argument("--method", type=str, choices=["llm", "constructive", "improved"],
                        default="llm", help="Generation method to use")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Check if the OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY") and args.method == "llm":
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set it with your OpenAI API key before running this script.")
        print("Alternatively, use --method=constructive to use the constructive generator without an API key.")
        sys.exit(1)
    
    print(f"Generating palindromic paragraphs...")
    print(f"Method: {args.method}")
    if args.method == "llm":
        print(f"Using model: {args.model}")
        print(f"Using NLTK seed: {args.use_nltk_seed}")
    print(f"Minimum length: {args.min_length} characters")
    print(f"Maximum attempts: {args.attempts}")
    
    # Choose the generator based on the method
    if args.method == "improved" or args.use_improved:
        # Use the improved generator
        try:
            from improved_generator import ImprovedPalindromeGenerator
            
            print("\nUsing improved multi-sentence palindrome generator...")
            generator = ImprovedPalindromeGenerator()
            palindrome, normalized, is_valid = generator.generate_multi_sentence_palindrome(
                min_length=args.min_length,
                attempts=args.attempts,
                model=args.model
            )
            
            if is_valid:
                show_palindrome_info(palindrome, normalized)
            else:
                print("\n⚠️ Improved generator failed to generate a valid palindrome.")
                print(f"Best attempt: {palindrome}")
                
        except Exception as e:
            print(f"Error using improved generator: {str(e)}")
            print("Falling back to enhanced generation...")
            args.method = "llm"  # Fall back to LLM method
    
    if args.method == "constructive":
        # Use the constructive generator
        print("\nUsing constructive palindrome generator...")
        palindrome, normalized, is_valid = enhanced_palindrome_generation(
            attempts=args.attempts,
            min_length=args.min_length,
            use_nltk_seed=args.use_nltk_seed,
            model=args.model,
            method="constructive"
        )
        
        show_palindrome_info(palindrome, normalized)
    
    elif args.method == "llm":
        # Use the LLM-based generator
        print("\nUsing LLM-based palindrome generator...")
        palindrome, normalized, is_valid = enhanced_palindrome_generation(
            attempts=args.attempts,
            min_length=args.min_length,
            use_nltk_seed=args.use_nltk_seed,
            model=args.model,
            method="llm"
        )
        
        show_palindrome_info(palindrome, normalized)

if __name__ == "__main__":
    main()
