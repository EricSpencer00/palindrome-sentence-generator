import os
import sys
import time
from dotenv import load_dotenv
from utils import generate_palindromic_paragraph
from validator import is_full_palindrome, normalize

def main():
    # Load environment variables
    load_dotenv()
    
    # Check if the OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set it with your OpenAI API key before running this script.")
        sys.exit(1)
    
    print("Generating palindromic paragraphs using Google Gemma model via OpenRouter...")
    
    max_attempts = 5
    success = False
    
    for i in range(max_attempts):
        try:
            print(f"\nAttempt {i+1}...")
            # Use the simplified prompt
            paragraph = generate_palindromic_paragraph("prompts/simple_palindrome.txt")
            
            if not paragraph or paragraph.strip() == "":
                print("Received empty response. Trying again...")
                time.sleep(2)  # Add a small delay between attempts
                continue
                
            print(f"Generated text: {paragraph}")
            
            # Validate if it's a palindrome
            if is_full_palindrome(paragraph):
                print("\n✅ Valid full palindrome paragraph!")
                normalized = normalize(paragraph)
                print(f"\nNormalized length: {len(normalized)} characters")
                print(f"Normalized text: {normalized}")
                success = True
                break
            else:
                print("❌ Not a valid full-character palindrome.")
                normalized = normalize(paragraph)
                print(f"Normalized: {normalized}")
                if len(normalized) > 20:
                    # Check if it's close to being a palindrome
                    reversed_norm = normalized[::-1]
                    match_count = sum(1 for a, b in zip(normalized, reversed_norm) if a == b)
                    match_percentage = (match_count / len(normalized)) * 100
                    print(f"Match percentage: {match_percentage:.2f}%")
                
        except Exception as e:
            print(f"Error during generation attempt {i+1}: {str(e)}")
            time.sleep(3)  # Add a longer delay after an error
    
    if not success:
        print(f"\nFailed to generate a valid palindrome after {max_attempts} attempts.")

if __name__ == "__main__":
    main()
