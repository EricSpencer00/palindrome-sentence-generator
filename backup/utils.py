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

def generate_constructive_palindrome(min_length=60):
    """
    Generate a palindrome using a constructive approach - building from the middle out
    or from beginning to end with a clear structure.
    """
    # Start with a palindromic seed (either a word or a small phrase)
    palindrome_words = get_palindrome_words()
    seed_options = palindrome_words + ["a", "i", "wow", "noon", "deed", "level", "radar", "rotor"]
    
    # Pick a random seed word
    if random.random() < 0.3:
        # 30% chance to use a more complex core
        seed = random.choice(["a man a plan a canal panama", "madam im adam", "never odd or even"])
        # Make it a proper palindrome by mirroring it
        normalized = normalize(seed)
        if len(normalized) % 2 == 0:  # Even length
            half = len(normalized) // 2
            normalized = normalized[:half] + normalized[half-1::-1]
        else:  # Odd length
            half = len(normalized) // 2
            normalized = normalized[:half+1] + normalized[half-1::-1]
    else:
        # 70% chance to use a simple palindrome word as seed
        seed = random.choice(seed_options)
        normalized = normalize(seed)
    
    # Build outward until we reach the minimum length
    while len(normalized) < min_length:
        # Strategies for extending the palindrome
        strategy = random.choice(["mirror", "letter_pair", "word_pair"])
        
        if strategy == "mirror" or len(normalized) > min_length * 0.8:
            # Simply add a random letter to both ends
            letter = chr(random.randint(97, 122))  # a-z
            normalized = letter + normalized + letter
        
        elif strategy == "letter_pair":
            # Add a small palindromic sequence (2-3 letters)
            options = ["aa", "bb", "cc", "dd", "ee", "aba", "aca", "ada", "aha", "ala", "ama", "ana", "apa", "ara", "ata", "ava"]
            pair = random.choice(options)
            # Randomly decide to add to beginning or end
            if random.random() < 0.5:
                normalized = pair + normalized + pair[::-1]
            else:
                # Choose a position to insert (not at the very beginning or end)
                if len(normalized) > 10:
                    pos = random.randint(1, len(normalized) // 2 - 1)
                    normalized = normalized[:pos] + pair + normalized[pos:len(normalized)-pos] + pair[::-1] + normalized[len(normalized)-pos:]
                else:
                    normalized = pair + normalized + pair[::-1]
        
        elif strategy == "word_pair":
            # Add a pair of words that are palindromic when combined
            palindromic_pairs = get_palindromic_pairs()
            if palindromic_pairs:
                pair = random.choice(palindromic_pairs)
                normalized = normalize(pair[0]) + normalized + normalize(pair[1])
            else:
                # Fallback to adding random letters
                letter = chr(random.randint(97, 122))  # a-z
                normalized = letter + normalized + letter
    
    # Convert normalized text back to readable form with proper spacing and capitalization
    return convert_to_readable_text(normalized)

def convert_to_readable_text(normalized_text):
    """
    Convert a normalized palindrome text (lowercase, no spaces/punctuation) 
    into readable text with proper spacing, capitalization, and punctuation.
    """
    # First, identify potential word boundaries
    common_words = set()
    
    # Add common short words that appear in palindromes
    short_words = ["a", "i", "am", "an", "as", "at", "be", "by", "do", "go", "he", "hi", "if", "in", 
                  "is", "it", "me", "my", "no", "of", "on", "or", "so", "to", "up", "us", "we"]
    common_words.update(short_words)
    
    # Add common palindrome words
    palindrome_words = ["mom", "dad", "wow", "eye", "pop", "tot", "bob", "sis", "noon", "deed", 
                       "peep", "sees", "madam", "level", "radar", "refer", "civic", "kayak", "solos"]
    common_words.update(palindrome_words)
    
    # Try to extract english words from normalized text
    try:
        from nltk.corpus import words as nltk_words
        english_words = set(w.lower() for w in nltk_words.words() if 2 <= len(w) <= 8)
        common_words.update(english_words)
    except Exception as e:
        print(f"Warning: Error accessing NLTK words: {e}")
        pass  # Continue without NLTK words if there's an issue
    
    # Build the text by identifying words
    result = ""
    i = 0
    
    while i < len(normalized_text):
        # Try to find the longest matching word at this position
        found_word = False
        for length in range(min(8, len(normalized_text) - i), 1, -1):
            potential_word = normalized_text[i:i+length]
            if potential_word in common_words:
                result += " " + potential_word
                i += length
                found_word = True
                break
        
        if not found_word:
            # If no word was found, add the character and move on
            if not result:  # Start of string
                result += normalized_text[i]
            else:
                # Sometimes group 2-3 letters together to create pronounceable chunks
                if random.random() < 0.3 and i + 2 <= len(normalized_text):
                    result += " " + normalized_text[i:i+2]
                    i += 2
                else:
                    result += " " + normalized_text[i]
                    i += 1
    
    # Clean up the text with proper capitalization and punctuation
    result = result.strip()
    sentences = []
    words = result.split()
    
    current_sentence = []
    for i, word in enumerate(words):
        current_sentence.append(word)
        
        # End sentence at random intervals or at the end
        if (i > 3 and random.random() < 0.2) or i == len(words) - 1:
            if current_sentence:
                # Capitalize first word
                current_sentence[0] = current_sentence[0].capitalize()
                
                # Add punctuation based on sentence content
                sentence = " ".join(current_sentence)
                if "?" in sentence or random.random() < 0.2:
                    sentence += "?"
                else:
                    sentence += "."
                
                sentences.append(sentence)
                current_sentence = []
    
    return " ".join(sentences)

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

def get_palindrome_words(min_length=3, max_length=8):
    """Get a list of palindrome words from NLTK corpus"""
    palindromes = []
    
    try:
        english_words = set(w.lower() for w in words.words() if min_length <= len(w) <= max_length)
        # Check each word to see if it's a palindrome
        for word in english_words:
            if word == word[::-1]:
                palindromes.append(word)
    except Exception as e:
        print(f"Warning: Error accessing NLTK words: {e}")
        # Fallback to common palindromes
        palindromes = ["mom", "dad", "wow", "eye", "pop", "sis", "noon", "level", 
                      "radar", "civic", "kayak", "refer", "madam", "rotor", "stats"]
    
    return palindromes

def get_palindromic_pairs():
    """Get pairs of words that form palindromes when combined"""
    pairs = []
    
    # Manual common pairs
    manual_pairs = [
        ("no", "on"), 
        ("was", "saw"), 
        ("draw", "ward"),
        ("live", "evil"), 
        ("stop", "pots"),
        ("time", "emit"),
        ("star", "rats"),
        ("flow", "wolf")
    ]
    pairs.extend(manual_pairs)
    
    # Try to find some pairs from NLTK
    try:
        # Get a sample of short words
        short_words = [w.lower() for w in words.words() if 2 <= len(w) <= 5]
        sample_size = min(1000, len(short_words))
        word_sample = random.sample(short_words, sample_size)
        
        # Check for reversible pairs
        for word in word_sample:
            reversed_word = word[::-1]
            if reversed_word in short_words and reversed_word != word:
                pairs.append((word, reversed_word))
    except Exception as e:
        print(f"Warning: Error finding palindromic pairs: {e}")
    
    return pairs

def generate_palindrome_seed(min_length=20):
    """Generate a palindromic seed phrase to start with"""
    palindrome_words = get_palindrome_words()
    
    # Methods to generate the seed
    methods = ["mirror", "center_out", "predefined"]
    method = random.choice(methods)
    
    if method == "mirror":
        # Create a palindrome by mirroring words
        words_to_use = random.randint(3, 6)
        seed_words = []
        
        for _ in range(words_to_use):
            if random.random() < 0.3 and palindrome_words:
                # Use a palindrome word occasionally
                seed_words.append(random.choice(palindrome_words))
            else:
                # Use common short words
                common_words = ["a", "i", "am", "an", "as", "at", "be", "by", "do", "go", 
                               "he", "hi", "if", "in", "is", "it", "me", "my", "no", "of", 
                               "on", "or", "so", "to", "up", "us", "we"]
                seed_words.append(random.choice(common_words))
        
        # Mirror the words to create a palindrome
        seed_phrase = " ".join(seed_words)
        normalized = normalize(seed_phrase)
        
        # If it's an even length, mirror exactly
        if len(normalized) % 2 == 0:
            mirrored = normalized + normalized[::-1]
        else:
            # For odd length, avoid duplicating the middle character
            mirrored = normalized + normalized[:-1][::-1]
        
        # Convert back to readable text with proper spacing
        return convert_to_readable_text(mirrored)
    
    elif method == "center_out":
        # Start with a palindrome word in the center and build outward
        if palindrome_words:
            center = random.choice(palindrome_words)
        else:
            center = random.choice(["radar", "level", "civic", "madam", "noon"])
        
        normalized = normalize(center)
        
        # Build outward until we reach the minimum length
        while len(normalized) < min_length:
            # Add a letter or short sequence to both ends
            if random.random() < 0.7:
                # Add a single letter
                letter = chr(random.randint(97, 122))  # a-z
                normalized = letter + normalized + letter
            else:
                # Add a small palindromic sequence
                options = ["aba", "aca", "ada", "aea", "aha", "ala", "ama", "ana", "apa", "ara", "asa", "ata", "ava"]
                seq = random.choice(options)
                normalized = seq + normalized + seq[::-1]
        
        return convert_to_readable_text(normalized)
    
    else:  # predefined
        # Use a predefined palindromic phrase
        phrases = [
            "a man a plan a canal panama",
            "never odd or even",
            "rats live on no evil star",
            "live not on evil",
            "madam im adam",
            "step on no pets",
            "was it a rat i saw",
            "no lemon no melon",
            "evil is a name of a foeman as i live"
        ]
        phrase = random.choice(phrases)
        
        # Ensure it's actually a palindrome
        normalized = normalize(phrase)
        if normalized != normalized[::-1]:
            # Fix it to be a palindrome
            half_length = len(normalized) // 2
            if len(normalized) % 2 == 0:
                normalized = normalized[:half_length] + normalized[:half_length][::-1]
            else:
                normalized = normalized[:half_length+1] + normalized[:half_length][::-1]
        
        # If it's too short, extend it
        while len(normalized) < min_length:
            letter = chr(random.randint(97, 122))  # a-z
            normalized = letter + normalized + letter
        
        return convert_to_readable_text(normalized)
