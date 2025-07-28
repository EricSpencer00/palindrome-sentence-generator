#!/usr/bin/env python3
"""
Palindrome Sentence Generator
This script generates sentences that are palindromes (read the same forwards and backwards)
and attempts to make them grammatically correct or sensible.
"""

import random
import re
import argparse

# Word lists categorized by parts of speech
NOUNS = [
    "mom", "dad", "eye", "level", "madam", "noon", "race car", "radar", "refer", "civic",
    "deed", "kayak", "level", "madam", "mom", "noon", "racecar", "radar", "refer", "rotor",
    "solos", "stats", "tenet", "wow"
]

VERBS = [
    "did", "sees", "was", "nod", "draw", "won", "eye", "poop", "swap", "rotator",
    "gag", "pull up", "put up", "step on", "live on", "lay", "pot", "tip", "tap"
]

ADJECTIVES = [
    "civic", "level", "madam", "noon", "radar", "rotor", "solos", "stats", "tenet", "wow", 
    "deified", "racecar", "repaper", "reviver", "rotator", "never odd or even"
]

ADVERBS = [
    "now", "on", "so", "too"
]

PREPOSITIONS = [
    "on", "by", "no", "to"
]

PALINDROME_PHRASES = [
    "a man a plan a canal panama",
    "never odd or even",
    "no lemon no melon",
    "live not on evil",
    "step on no pets",
    "was it a car or a cat i saw",
    "go hang a salami im a lasagna hog",
    "do geese see god",
    "rats live on no evil star",
    "murder for a jar of red rum",
    "some men interpret nine memos",
    "doc note i dissent a fast never prevents a fatness i diet on cod",
    "too bad i hid a boot",
    "no sir prefer prison",
    "never odd or even",
    "race fast safe car",
    "draw o coward",
    "pull a bat i bite pill up",
    "evil is a name of a foeman as i live",
    "taco cat",
]

# Lists of words that are used as building blocks for creating palindromes
PALINDROME_WORDS = NOUNS + VERBS + ADJECTIVES + ADVERBS + PREPOSITIONS

def clean_text(text):
    """Remove spaces and punctuation, convert to lowercase"""
    return re.sub(r'[^a-z]', '', text.lower())

def is_palindrome(text):
    """Check if text is a palindrome"""
    cleaned = clean_text(text)
    return cleaned == cleaned[::-1]

def reverse_text(text):
    """Reverse a string maintaining spaces and punctuation"""
    # Remove punctuation and spaces, then reverse
    cleaned = clean_text(text)
    reversed_cleaned = cleaned[::-1]
    
    # Reinsert spaces and punctuation
    result = ""
    clean_idx = 0
    
    for char in text:
        if char.isalpha():
            result += reversed_cleaned[clean_idx]
            clean_idx += 1
        else:
            result += char
    
    return result

def generate_random_palindrome():
    """Generate a random palindrome from pre-defined phrases"""
    return random.choice(PALINDROME_PHRASES)

def generate_mirrored_palindrome():
    """Generate a palindrome by mirroring words that are themselves palindromes"""
    # Choose random number of words for the first half (1 to 3)
    num_words = random.randint(1, 3)
    
    # Create first half of the palindrome using only palindrome words
    # Filter to only use words that are themselves palindromes
    palindrome_only_words = [word for word in PALINDROME_WORDS if is_palindrome(word)]
    
    words = []
    for _ in range(num_words):
        words.append(random.choice(palindrome_only_words))
    
    first_half = " ".join(words)
    
    # Choose a center word (optional) that must be a palindrome
    has_center = random.choice([True, False])
    center_word = ""
    
    if has_center:
        center_word = random.choice(palindrome_only_words)
    
    # Create second half by reversing the first half
    second_half = " ".join(words[::-1])
    
    # Combine parts to create the palindrome
    if has_center:
        palindrome = f"{first_half} {center_word} {second_half}"
    else:
        palindrome = f"{first_half} {second_half}"
    
    return palindrome

def build_structured_palindrome():
    """Generate a more structured palindrome sentence with better grammar"""
    structures = [
        # Subject + Verb structures
        lambda: f"{random.choice(NOUNS)} {random.choice(VERBS)}",
        
        # Subject + Verb + Object structures
        lambda: f"{random.choice(NOUNS)} {random.choice(VERBS)} {random.choice(NOUNS)}",
        
        # Adjective + Noun structures
        lambda: f"{random.choice(ADJECTIVES)} {random.choice(NOUNS)}",
        
        # Prepositional phrases
        lambda: f"{random.choice(PREPOSITIONS)} {random.choice(NOUNS)}",
        
        # Subject + Verb + Prepositional Phrase
        lambda: f"{random.choice(NOUNS)} {random.choice(VERBS)} {random.choice(PREPOSITIONS)} {random.choice(NOUNS)}"
    ]
    
    # Select a random structure
    structure_func = random.choice(structures)
    first_half = structure_func()
    
    # Create a center word/phrase (optional)
    has_center = random.choice([True, False])
    center = ""
    if has_center:
        center = random.choice(PALINDROME_WORDS)
    
    # Create second half by reversing words and their order
    words = first_half.split()
    reversed_words = []
    
    for word in reversed(words):
        reversed_words.append(word[::-1])
    
    second_half = " ".join(reversed_words)
    
    # Combine to create the palindrome
    if has_center:
        return f"{first_half} {center} {second_half}"
    else:
        return f"{first_half} {second_half}"

def ensure_palindrome(text):
    """Ensure the text is a palindrome, with minor fixes if needed"""
    if is_palindrome(text):
        return text
    
    # Simple fixes for close-to-palindrome texts
    cleaned = clean_text(text)
    reversed_cleaned = cleaned[::-1]
    
    if cleaned.startswith(reversed_cleaned[:-1]):
        # Missing last character
        return text + text[0]
    
    # If it can't be easily fixed, use a pre-defined palindrome
    return generate_random_palindrome()

def generate_palindrome(method="random"):
    """Generate a palindrome using the specified method"""
    if method == "random":
        return generate_random_palindrome()
    elif method == "mirrored":
        return generate_mirrored_palindrome()
    elif method == "structured":
        return build_structured_palindrome()
    else:
        # Default to random
        return generate_random_palindrome()

def main():
    """Main function to run the palindrome generator"""
    parser = argparse.ArgumentParser(description='Generate palindrome sentences')
    parser.add_argument('--method', choices=['random', 'mirrored', 'structured'], 
                        default='random', help='Method to generate palindromes')
    parser.add_argument('--count', type=int, default=1, 
                        help='Number of palindromes to generate')
    args = parser.parse_args()
    
    for _ in range(args.count):
        palindrome = generate_palindrome(args.method)
        print(f"Palindrome: {palindrome}")
        
        # Verify it's a palindrome
        is_pal = is_palindrome(palindrome)
        print(f"Is palindrome: {is_pal}")
        
        if not is_pal:
            print("Warning: Generated text is not a perfect palindrome.")
        
        print()

if __name__ == "__main__":
    main()
