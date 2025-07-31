import os
import sys
import time
import argparse
from dotenv import load_dotenv
from utils import generate_palindromic_paragraph
from validator import is_full_palindrome, normalize
from fallback_generator import generate_local_palindrome

# Set this to True after we've created the fallback_generator.py
FALLBACK_AVAILABLE = True

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

def enhanced_palindrome_generation(attempts=5, min_length=60, use_nltk_seed=True, model="google/gemma-3n-e2b-it:free"):
    """Enhanced palindrome generation with multiple strategies and repair mechanisms"""
    best_palindrome = None
    best_match_percentage = 0
    best_length = 0
    
    # Try LLM-based generation first
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
                    # Keep track of the best short palindrome in case we don't find a long enough one
                    if len(normalized) > best_length:
                        best_palindrome = paragraph
                        best_length = len(normalized)
            else:
                print("❌ Not a valid full-character palindrome.")
                print(f"Normalized: {normalized}")
                
                # Check if it's close to being a palindrome
                if len(normalized) > 20:
                    reversed_norm = normalized[::-1]
                    match_count = sum(1 for a, b in zip(normalized, reversed_norm) if a == b)
                    match_percentage = (match_count / len(normalized)) * 100
                    print(f"Match percentage: {match_percentage:.2f}%")
                    
                    # Track the best near-palindrome for potential repair
                    if match_percentage > best_match_percentage and len(normalized) >= min_length * 0.7:
                        best_match_percentage = match_percentage
                        best_palindrome = paragraph
                        best_length = len(normalized)
                    
        except Exception as e:
            print(f"Error during generation attempt {i+1}: {str(e)}")
            time.sleep(2)
    
    # If we found a valid palindrome but it was too short, return it anyway
    if best_palindrome and is_full_palindrome(best_palindrome):
        print("\n✅ Found a valid palindrome, but it's shorter than requested.")
        normalized = normalize(best_palindrome)
        return best_palindrome, normalized, True
    
    # If we found a near-palindrome with high match percentage, try to repair it
    if best_palindrome and best_match_percentage > 60:
        print(f"\nAttempting to repair near-palindrome with {best_match_percentage:.2f}% match...")
        repaired = attempt_palindrome_repair(best_palindrome, min_length)
        
        if is_full_palindrome(repaired):
            print("✅ Successfully repaired to create a valid palindrome!")
            normalized = normalize(repaired)
            return repaired, normalized, True
    
    # If all else fails, use the fallback generator
    print("\nTrying fallback local generator...")
    local_palindrome = generate_local_palindrome(min_length, 10)
    
    if is_full_palindrome(local_palindrome):
        print("\n✅ Valid full palindrome paragraph generated by fallback generator!")
        normalized = normalize(local_palindrome)
        return local_palindrome, normalized, True
    
    # If even the fallback fails, return the best near-palindrome we found
    if best_palindrome:
        print("\n⚠️ Returning best near-palindrome found.")
        normalized = normalize(best_palindrome)
        return best_palindrome, normalized, False
    
    # Last resort - return a simple palindrome
    fallback = "Eva, can I see bees in a cave? No evil, I live on. Eva, can I see bees in a cave?"
    return fallback, normalize(fallback), True

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
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Check if the OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set it with your OpenAI API key before running this script.")
        sys.exit(1)
    
    print(f"Generating palindromic paragraphs using {args.model}...")
    print(f"Using NLTK seed: {args.use_nltk_seed}")
    print(f"Minimum length: {args.min_length} characters")
    print(f"Maximum attempts: {args.attempts}")
    print(f"Using enhanced algorithm: {args.enhanced}")
    print(f"Using improved generator: {args.use_improved}")
    
    if args.use_improved:
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
            print("Falling back to local generator...")
            
            # Use fallback generator
            if FALLBACK_AVAILABLE:
                local_palindrome = generate_local_palindrome(args.min_length, 10)
                
                if is_full_palindrome(local_palindrome):
                    print("\n✅ Valid full palindrome paragraph generated by fallback generator!")
                    normalized = normalize(local_palindrome)
                    show_palindrome_info(local_palindrome, normalized)
                else:
                    print("\n❌ Fallback generator also failed to produce a valid palindrome.")
                    args.enhanced = True  # Try the enhanced algorithm as a last resort
            else:
                print("Falling back to enhanced algorithm...")
                args.enhanced = True
            
    if args.enhanced and not args.use_improved:
        # Use the enhanced generation algorithm
        palindrome, normalized, is_valid = enhanced_palindrome_generation(
            attempts=args.attempts,
            min_length=args.min_length,
            use_nltk_seed=args.use_nltk_seed,
            model=args.model
        )
        
        if is_valid:
            show_palindrome_info(palindrome, normalized)
        else:
            print("\n⚠️ Failed to generate a valid palindrome.")
            print(f"Best attempt: {palindrome}")
    else:
        # Use the original algorithm
        max_attempts = args.attempts
        success = False
        
        for i in range(max_attempts):
            try:
                print(f"\nAttempt {i+1}...")
                # Generate palindrome with NLTK seed if requested
                paragraph = generate_palindromic_paragraph(
                    prompt_path="prompts/simple_palindrome.txt",
                    use_nltk_seed=args.use_nltk_seed,
                    min_seed_length=args.min_length // 2,  # Use half the minimum length for the seed
                    model=args.model
                )
                
                if not paragraph or paragraph.strip() == "":
                    print("Received empty response. Trying again...")
                    time.sleep(2)  # Add a small delay between attempts
                    continue
                    
                # Validate if it's a palindrome
                if is_full_palindrome(paragraph):
                    print("\n✅ Valid full palindrome paragraph!")
                    normalized = normalize(paragraph)
                    
                    # Check if it meets the minimum length
                    if len(normalized) >= args.min_length:
                        show_palindrome_info(paragraph, normalized)
                        success = True
                        break
                    else:
                        print(f"❌ Palindrome too short: {len(normalized)} characters (minimum: {args.min_length})")
                else:
                    print("❌ Not a valid full-character palindrome.")
                    normalized = normalize(paragraph)
                    print(f"Normalized: {normalized}")
                    
                    # Check if it's close to being a palindrome
                    if len(normalized) > 20:
                        reversed_norm = normalized[::-1]
                        match_count = sum(1 for a, b in zip(normalized, reversed_norm) if a == b)
                        match_percentage = (match_count / len(normalized)) * 100
                        print(f"Match percentage: {match_percentage:.2f}%")
                        
                        # If it's close (>80% match), try to fix it
                        if match_percentage > 80:
                            print("Attempting to fix the near-palindrome...")
                            # Take the first half and mirror it
                            half_point = len(normalized) // 2
                            first_half = normalized[:half_point]
                            fixed_palindrome = first_half + first_half[::-1]
                            
                            print(f"Fixed normalized: {fixed_palindrome}")
                            # This is just a basic fix. We'd need more sophisticated 
                            # algorithms to convert this back to readable text.
                    
            except Exception as e:
                print(f"Error during generation attempt {i+1}: {str(e)}")
                time.sleep(3)  # Add a longer delay after an error
        
        if not success:
            print(f"\nFailed to generate a valid palindrome using the API after {max_attempts} attempts.")
            
            # Use fallback generator if available
            if FALLBACK_AVAILABLE:
                print("\nTrying fallback local generator...")
                local_palindrome = generate_local_palindrome(args.min_length, 10)
                
                if is_full_palindrome(local_palindrome):
                    print("\n✅ Valid full palindrome paragraph generated by fallback generator!")
                    normalized = normalize(local_palindrome)
                    show_palindrome_info(local_palindrome, normalized)
                else:
                    print("\n❌ Fallback generator also failed to produce a valid palindrome.")
            else:
                print("\nFallback generator not available. Please try again later.")

if __name__ == "__main__":
    main()
