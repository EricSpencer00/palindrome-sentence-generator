import os
import random
import re
import nltk
from nltk.corpus import words, wordnet
from nltk.tokenize import word_tokenize
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file (if it exists)
load_dotenv()

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/words')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading required NLTK resources...")
    nltk.download('punkt')
    nltk.download('words')
    nltk.download('wordnet')

# Get the API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Check if we're using an OpenRouter API key (typically starts with 'sk-or')
is_openrouter = api_key.startswith('sk-or')

# Initialize the OpenAI client with appropriate base URL if using OpenRouter
if is_openrouter:
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
else:
    client = OpenAI(api_key=api_key)

def get_palindrome_words():
    """Get a list of palindrome words from NLTK corpus"""
    english_words = set(words.words())
    palindromes = [word.lower() for word in english_words 
                  if len(word) > 2 and word.lower() == word.lower()[::-1]]
    return palindromes

def get_palindromic_pairs():
    """Get pairs of words that can form palindromes when combined"""
    english_words = set(w.lower() for w in words.words() if len(w) >= 3)
    pairs = []
    
    for word in english_words:
        reversed_word = word[::-1]
        if reversed_word in english_words and reversed_word != word:
            pairs.append((word, reversed_word))
    
    return pairs

def generate_palindrome_seed(min_length=20):
    """Generate a palindrome seed using NLTK resources"""
    # Get palindromic words and pairs
    palindrome_words = get_palindrome_words()
    palindromic_pairs = get_palindromic_pairs()
    
    # Start with a palindromic word or empty string
    if random.random() < 0.5 and palindrome_words:
        seed = random.choice(palindrome_words)
    else:
        seed = ""
    
    # Add palindromic pairs until we reach the minimum length
    while len(seed) < min_length:
        if palindromic_pairs:
            pair = random.choice(palindromic_pairs)
            if random.random() < 0.5:
                # Add pair to both sides
                seed = pair[0] + seed + pair[1]
            else:
                # Add reversed pair
                seed = pair[1] + seed + pair[0]
        else:
            # If no pairs available, add random letters
            letter = chr(random.randint(97, 122))  # a-z
            seed = letter + seed + letter
    
    return seed

def generate_palindromic_examples():
    """Generate examples of valid palindromes to guide the LLM"""
    examples = [
        "Eva, can I see bees in a cave? No evil, I live on. Eva, can I see bees in a cave?",
        "A man, a plan, a canal: Panama. A man, a plan, a canal: Panama.",
        "Madam, I'm Adam. Madam, I'm Adam.",
        "Step on no pets. Step on no pets."
    ]
    
    # Add some palindrome examples with their normalized forms
    formatted_examples = []
    for example in examples:
        norm = normalize(example)
        formatted_examples.append(f"Example: '{example}' → Normalized: {norm}")
    
    return "\n".join(formatted_examples)

def generate_palindrome_strategies():
    """Generate strategy tips for creating palindromes"""
    strategies = [
        "1. Start with a central palindromic word like 'noon', 'level', or 'radar'",
        "2. Build outward symmetrically, adding the same letters or words to both sides",
        "3. Use common palindromic phrases as building blocks",
        "4. Remember that only letters matter, not spaces or punctuation",
        "5. Create the first half and then mirror it exactly for the second half"
    ]
    return "\n".join(strategies)

def normalize(text):
    """Normalize text by removing non-alphabetic characters and converting to lowercase"""
    return re.sub(r'[^a-zA-Z]', '', text).lower()

def generate_palindromic_paragraph(prompt_path="prompts/full_paragraph.txt", 
                                  use_nltk_seed=True,
                                  min_seed_length=20,
                                  model="google/gemma-3n-e2b-it:free"):
    """Generate a palindromic paragraph using both NLTK and LLM"""
    with open(prompt_path, "r") as f:
        prompt = f.read()
    
    # Generate a high-quality seed palindrome
    seed = generate_palindrome_seed(min_seed_length) if use_nltk_seed else ""
    
    # Create an enhanced prompt with examples, strategies, and seed
    examples = generate_palindromic_examples()
    strategies = generate_palindrome_strategies()
    
    enhanced_prompt = f"""{prompt}

Here are some valid palindrome examples:
{examples}

Palindrome creation strategies:
{strategies}

Remember that for a FULL palindrome, the entire text (when normalized by removing spaces, punctuation, and standardizing case) should read the same forward and backward.

Start with this palindromic seed: '{seed}'

Your palindrome must be at least 60 characters long when normalized.

IMPORTANT: The ENTIRE text must be a SINGLE palindrome when all spaces, punctuation, and capitalization are removed - not multiple separate palindromes. Do not create multiple small palindromes; create ONE large palindrome for the entire text.

I will validate your result by removing all spaces and punctuation and checking if it reads the same forward and backward. Double-check your work before submitting.
"""
    
    # Use LLM to generate the palindromic paragraph with a focused approach
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": enhanced_prompt}],
            temperature=0.7,
            max_tokens=300,
            timeout=60
        )
        
        generated_text = response.choices[0].message.content.strip()
        
        # If the generated text contains explanations or notes, try to extract just the palindrome
        if len(generated_text.split("\n")) > 3:
            # Look for the longest line which is likely the palindrome
            lines = [line.strip() for line in generated_text.split("\n") if line.strip()]
            longest_line = max(lines, key=len)
            if len(longest_line) > 20:  # Assume it's the palindrome if reasonably long
                return longest_line
        
        return generated_text
    
    except Exception as e:
        print(f"Error during LLM generation: {str(e)}")
        # Return the seed as a fallback if LLM fails
        return seed
