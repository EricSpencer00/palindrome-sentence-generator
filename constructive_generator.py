#!/usr/bin/env python3
# filepath: /Users/ericspencer/GitHub/palindrome-sentence-generator/constructive_generator.py
"""
Constructive Palindrome Generator

This module implements a constructive approach to palindrome generation,
inspired by Roland Meertens' approach described in his paper on creating
the "most complete palindromic sentence".

Key techniques:
1. Building palindromes from the middle out
2. Combining smaller palindromic units
3. Using word pairs that form palindromes when combined
4. Split-and-mirror approach for creating longer palindromes
"""

import random
import re
import nltk
from nltk.corpus import words, wordnet
from validator import normalize

# Ensure NLTK resources are available
try:
    nltk.data.find('corpora/words')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('words')
    nltk.download('wordnet')

class ConstructivePalindromeGenerator:
    """
    A class that implements constructive approaches to palindrome generation,
    focusing on building palindromes from smaller units or from the middle out.
    """
    
    def __init__(self):
        """Initialize the generator with common palindrome words and reversible pairs."""
        self.palindrome_words = self._get_palindrome_words()
        self.reversible_pairs = self._get_palindromic_pairs()
        self.common_words = {
            'a', 'i', 'am', 'an', 'as', 'at', 'be', 'by', 'do', 'go', 'he', 'hi', 
            'if', 'in', 'is', 'it', 'me', 'my', 'no', 'of', 'on', 'or', 'so', 
            'to', 'up', 'us', 'we'
        }
        
        # Add some common palindromic phrases
        self.palindrome_phrases = [
            "a man a plan a canal panama",
            "madam im adam",
            "never odd or even",
            "rats live on no evil star",
            "step on no pets",
            "was it a car or a cat i saw",
            "no lemon no melon"
        ]
        
    def _get_palindrome_words(self, min_length=3, max_length=8):
        """Get a list of palindrome words from NLTK corpus."""
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
        
        # Add some extra common palindromes that might not be in NLTK
        extra_palindromes = [
            "deed", "peep", "poop", "boob", "toot", "racecar", "rotator", "deified",
            "redder", "reviver", "redivider", "detartrated", "tattarrattat"
        ]
        
        for word in extra_palindromes:
            if word not in palindromes:
                palindromes.append(word)
        
        return palindromes
    
    def _get_palindromic_pairs(self):
        """Get pairs of words that form palindromes when combined."""
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
            ("flow", "wolf"),
            ("rail", "liar"),
            ("taco", "cat"),
            ("god", "dog"),
            ("mood", "doom"),
            ("mug", "gum"),
            ("ton", "not"),
            ("pot", "top"),
            ("pool", "loop"),
            ("part", "trap"),
            ("war", "raw"),
            ("reward", "drawer"),
            ("net", "ten"),
            ("wolf", "flow"),
            ("stressed", "desserts"),
            ("night", "thin"),
            ("nail", "lian"),
            ("smart", "trams"),
            ("step", "pets"),
            ("deliver", "reviled"),
            ("noon", "noon"),
            ("level", "level"),
            ("sleep", "peels"),
            ("keep", "peek"),
            ("straw", "warts"),
            ("lived", "devil"),
            ("den", "ned"),
            ("now", "won"),
            ("pans", "snap"),
            ("stun", "nuts"),
            ("wonk", "know"),
            ("pals", "slap"),
            ("spot", "tops"),
            ("pat", "tap"),
            ("pins", "snip"),
            ("port", "trop"),
            ("ward", "draw"),
            ("name", "eman"),
            ("brag", "garb"),
            ("maps", "spam"),
            ("laced", "decal"),
            ("tuba", "abut"),
            ("mark", "kram"),
            ("spin", "nips"),
            ("rats", "star")
        ]
        pairs.extend(manual_pairs)
        
        # Try to find some pairs from NLTK
        try:
            # Get a sample of short words
            short_words = [w.lower() for w in words.words() if 2 <= len(w) <= 5]
            sample_size = min(2000, len(short_words))
            word_sample = random.sample(short_words, sample_size)
            
            # Check for reversible pairs
            for word in word_sample:
                reversed_word = word[::-1]
                if reversed_word in short_words and reversed_word != word:
                    pairs.append((word, reversed_word))
        except Exception as e:
            print(f"Warning: Error finding palindromic pairs: {e}")
        
        return pairs
    
    def generate_constructive_palindrome(self, min_length=60):
        """
        Generate a palindrome using a constructive approach - building from the middle out
        or from beginning to end with a clear structure.
        """
        # Start with a palindromic seed (either a word or a small phrase)
        seed_options = self.palindrome_words + ["a", "i", "wow", "noon", "deed", "level", "radar", "rotor"]
        
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
                if self.reversible_pairs:
                    pair = random.choice(self.reversible_pairs)
                    normalized = normalize(pair[0]) + normalized + normalize(pair[1])
                else:
                    # Fallback to adding random letters
                    letter = chr(random.randint(97, 122))  # a-z
                    normalized = letter + normalized + letter
        
        # Convert normalized text back to readable form with proper spacing and capitalization
        return self._convert_to_readable_text(normalized)
    
    def _convert_to_readable_text(self, normalized_text):
        """
        Convert a normalized palindrome text (lowercase, no spaces/punctuation) 
        into readable text with proper spacing, capitalization, and punctuation.
        """
        # Build the text by identifying words
        result = ""
        i = 0
        
        # Add more common words that can be useful in palindromes
        common_words = self.common_words.union(set(self.palindrome_words))
        extra_words = {
            "a", "is", "as", "at", "be", "by", "do", "go", "he", "hi", "if", "in", "it", 
            "me", "my", "no", "of", "on", "or", "so", "to", "up", "us", "we", "yes",
            "the", "and", "but", "for", "not", "with", "this", "that", "have", "from",
            "one", "all", "was", "are", "were", "been", "has", "had", "can", "will",
            "would", "could", "should", "may", "might", "must", "shall", "part", "draw",
            "see", "saw", "seen", "gone", "done", "made", "said", "time", "more", "most",
            "some", "such", "than", "then", "them", "they", "there", "these", "those",
            "when", "where", "which", "while", "who", "whom", "whose", "why", "how",
            "now", "ever", "never", "always", "often", "soon", "today", "night", "day",
            "week", "month", "year", "here", "now", "back", "down", "over", "under",
            "again", "still", "yet", "too", "also", "well", "just", "even", "only",
            "very", "much", "more", "most", "other", "new", "old", "good", "bad",
            "high", "low", "big", "small", "long", "short", "far", "near", "fast",
            "slow", "hard", "soft", "hot", "cold", "wet", "dry", "full", "empty"
        }
        common_words.update(extra_words)
        
        # Try to add some more common words from NLTK
        try:
            from nltk.corpus import words as nltk_words
            english_words = set(w.lower() for w in nltk_words.words() if 2 <= len(w) <= 8)
            common_words.update(english_words)
        except Exception as e:
            print(f"Warning: Error accessing NLTK words: {e}")
        
        # First, try to find known palindrome words
        palindrome_positions = []
        for word in sorted(self.palindrome_words, key=len, reverse=True):
            pos = normalized_text.find(word)
            while pos >= 0:
                palindrome_positions.append((pos, pos + len(word), word))
                pos = normalized_text.find(word, pos + 1)
        
        # Sort by position
        palindrome_positions.sort()
        
        # Create segments to process, skipping the palindrome words
        segments = []
        last_end = 0
        for start, end, _ in palindrome_positions:
            if start > last_end:
                segments.append((last_end, start, normalized_text[last_end:start]))
            last_end = end
        if last_end < len(normalized_text):
            segments.append((last_end, len(normalized_text), normalized_text[last_end:]))
        
        # Now process each segment and the palindrome words
        all_parts = sorted(palindrome_positions + segments, key=lambda x: x[0])
        
        # Process each part
        words_in_result = []
        for start, end, text in all_parts:
            if isinstance(text, str) and len(text) > 1:
                # Process segment of text
                i = 0
                while i < len(text):
                    # Try to find the longest matching word at this position
                    found_word = False
                    for length in range(min(8, len(text) - i), 1, -1):
                        potential_word = text[i:i+length]
                        if potential_word in common_words:
                            words_in_result.append(potential_word)
                            i += length
                            found_word = True
                            break
                    
                    if not found_word:
                        # If no word was found, try to make pronounceable chunks
                        if i + 2 <= len(text) and random.random() < 0.6:
                            # Try to make 2-3 letter chunks that are pronounceable
                            chunk = text[i:i+min(3, len(text)-i)]
                            # Check if chunk has at least one vowel to be pronounceable
                            if any(c in "aeiou" for c in chunk):
                                words_in_result.append(chunk)
                                i += len(chunk)
                            else:
                                words_in_result.append(text[i])
                                i += 1
                        else:
                            words_in_result.append(text[i])
                            i += 1
            else:
                # Add palindrome word directly
                words_in_result.append(text)
        
        # Now convert the words into readable sentences
        current_sentence = []
        sentences = []
        
        for i, word in enumerate(words_in_result):
            current_sentence.append(word)
            
            # End sentence at random intervals or at the end
            if (len(current_sentence) > 3 and random.random() < 0.25) or i == len(words_in_result) - 1:
                if current_sentence:
                    # Capitalize first word
                    current_sentence[0] = current_sentence[0].capitalize()
                    
                    # Add punctuation based on sentence content
                    sentence = " ".join(current_sentence)
                    if random.random() < 0.1:
                        sentence += "?"
                    else:
                        sentence += "."
                    
                    sentences.append(sentence)
                    current_sentence = []
        
        return " ".join(sentences)
    
    def generate_center_out_palindrome(self, min_length=80):
        """
        Generate a palindrome by starting with a center word or letter and building outward.
        This approach tends to create more structured palindromes.
        """
        # Start with a center element
        if random.random() < 0.7:
            # Use a palindrome word
            center = random.choice(self.palindrome_words)
        else:
            # Use a single letter
            center = random.choice(list("aeiou"))  # Vowels make better centers
        
        normalized = normalize(center)
        
        # Add some famous palindrome patterns as potential centers
        famous_centers = [
            "amanaplanacanalpanama",  # A man, a plan, a canal: Panama
            "ablewasiereisawelba",    # Able was I ere I saw Elba
            "madam",
            "racecar",
            "neveroddoneven",         # Never odd or even
            "notlob",                 # Not a lob (from Monty Python)
            "ratsliveonnoevilstar",   # Rats live on no evil star
            "steponnopets",           # Step on no pets
            "wasitacatoracaratisaw"   # Was it a car or a cat I saw?
        ]
        
        if random.random() < 0.3:
            # Use a famous palindrome as center occasionally
            normalized = random.choice(famous_centers)
        
        # Build outward until we reach the minimum length
        while len(normalized) < min_length:
            # Choose how to extend
            extension_type = random.choices(
                ["single_letter", "word_pair", "palindrome_sequence", "common_pattern"],
                weights=[0.3, 0.4, 0.2, 0.1]
            )[0]
            
            if extension_type == "single_letter":
                # Add a single letter to both sides
                letter = random.choice(list("etaoinshrdlucmfwypvbgkjqxz"))  # Common English letters first
                normalized = letter + normalized + letter
                
            elif extension_type == "word_pair":
                if self.reversible_pairs:
                    # Add a reversible word pair
                    pair = random.choice(self.reversible_pairs)
                    normalized = normalize(pair[0]) + normalized + normalize(pair[1])
                else:
                    # Fallback to a single letter
                    letter = random.choice(list("etaoinshrdlucmfwypvbgkjqxz"))
                    normalized = letter + normalized + letter
            
            elif extension_type == "palindrome_sequence":
                # Add a palindromic sequence
                sequences = [
                    "aba", "aca", "ada", "aea", "aha", "ala", "ama", "ana", "apa", "ara", 
                    "asa", "ata", "ava", "awa", "axa", "aya", "aza", 
                    "ded", "did", "dod", "dud", 
                    "eve", "eye", 
                    "imi", "ini", 
                    "olo", "omo", "ono", "opo", "oso", "oto", "ovo", "owo", "oxo", 
                    "ullu", "ummu", "unnu", "uppu", "urru", "ussu", "uttu", "uvvu", 
                    "abcba", "acbca", "adbda", "aebea", "afbfa", "agbga", "ahbha"
                ]
                seq = random.choice(sequences)
                normalized = seq + normalized + seq
            
            else:  # common_pattern
                # Add a common palindromic pattern
                patterns = [
                    "aibohphobia",  # Fear of palindromes
                    "detartrated",
                    "rotavator",
                    "sensuousness",
                    "tattarrattat"
                ]
                pattern = normalize(random.choice(patterns))
                half_len = len(pattern) // 2
                if random.random() < 0.5:
                    # Add first half to beginning, second half to end
                    normalized = pattern[:half_len] + normalized + pattern[half_len:]
                else:
                    # Add whole pattern to one side (chosen randomly)
                    if random.random() < 0.5:
                        normalized = pattern + normalized
                    else:
                        normalized = normalized + pattern
        
        # Convert to readable text
        return self._convert_to_readable_text(normalized)
    
    def generate_mirrored_palindrome(self, min_length=80):
        """
        Generate a palindrome by creating the first half and then mirroring it.
        This tends to create symmetrically structured palindromes.
        """
        # Create the first half
        first_half = ""
        target_half_length = min_length // 2
        
        while len(first_half) < target_half_length:
            # Choose what to add
            addition_type = random.choices(
                ["common_word", "random_letters", "palindrome_word"],
                weights=[0.5, 0.3, 0.2]
            )[0]
            
            if addition_type == "common_word":
                word = random.choice(list(self.common_words))
                first_half += normalize(word)
                
            elif addition_type == "random_letters":
                # Add 1-3 random letters
                length = random.randint(1, 3)
                for _ in range(length):
                    first_half += random.choice(list("etaoinshrdlucmfwypvbgkjqxz"))
                    
            else:  # palindrome_word
                word = random.choice(self.palindrome_words)
                first_half += normalize(word)
        
        # Create the palindrome by mirroring
        normalized = first_half
        
        # Decide whether to make it odd or even length
        if random.random() < 0.5:  # Even length
            normalized = first_half + first_half[::-1]
        else:  # Odd length - add a middle character
            middle_char = random.choice(list("aeiou"))  # Vowels make better midpoints
            normalized = first_half + middle_char + first_half[::-1]
        
        # Convert to readable text
        return self._convert_to_readable_text(normalized)
    
    def generate_combinatorial_palindrome(self, min_length=100):
        """
        Generate a palindrome by combining multiple smaller palindromes.
        This approach can create longer, more varied palindromes.
        """
        # Start with an empty palindrome
        normalized = ""
        
        # Keep adding palindromic units until we reach the minimum length
        while len(normalized) < min_length:
            # Choose a palindromic unit to add
            unit_type = random.choices(
                ["phrase", "word", "mirror_letters", "reversible_pair"],
                weights=[0.1, 0.3, 0.3, 0.3]
            )[0]
            
            if unit_type == "phrase" and self.palindrome_phrases:
                # Add a palindromic phrase
                phrase = random.choice(self.palindrome_phrases)
                norm_phrase = normalize(phrase)
                
                # Make sure it's a proper palindrome
                if norm_phrase != norm_phrase[::-1]:
                    half = len(norm_phrase) // 2
                    if len(norm_phrase) % 2 == 0:
                        norm_phrase = norm_phrase[:half] + norm_phrase[:half][::-1]
                    else:
                        norm_phrase = norm_phrase[:half+1] + norm_phrase[:half][::-1]
                
                normalized += norm_phrase
                
            elif unit_type == "word":
                # Add a palindrome word
                word = random.choice(self.palindrome_words)
                normalized += normalize(word)
                
            elif unit_type == "mirror_letters":
                # Add a mirrored sequence of 2-4 letters
                length = random.randint(2, 4)
                letters = ''.join(random.choice(list("abcdefghijklmnopqrstuvwxyz")) for _ in range(length))
                normalized += letters + letters[::-1]
                
            else:  # reversible_pair
                if self.reversible_pairs:
                    # Add a reversible word pair
                    pair = random.choice(self.reversible_pairs)
                    normalized += normalize(pair[0]) + normalize(pair[1])
                else:
                    # Fallback to a palindrome word
                    word = random.choice(self.palindrome_words)
                    normalized += normalize(word)
        
        # Convert to readable text
        return self._convert_to_readable_text(normalized)

# Create a simple interface function to use with the existing system
def generate_constructive_palindrome(min_length=60, strategy="auto"):
    """
    Generate a palindrome using constructive approaches.
    
    Args:
        min_length: Minimum length of the normalized palindrome
        strategy: Which strategy to use - "auto", "constructive", "center_out", "mirrored", or "combinatorial"
    
    Returns:
        A palindromic text with proper spacing and capitalization
    """
    generator = ConstructivePalindromeGenerator()
    
    if strategy == "auto":
        # Choose a strategy randomly, weighted by effectiveness
        strategy = random.choices(
            ["constructive", "center_out", "mirrored", "combinatorial"],
            weights=[0.3, 0.3, 0.2, 0.2]
        )[0]
    
    if strategy == "constructive":
        return generator.generate_constructive_palindrome(min_length)
    elif strategy == "center_out":
        return generator.generate_center_out_palindrome(min_length)
    elif strategy == "mirrored":
        return generator.generate_mirrored_palindrome(min_length)
    elif strategy == "combinatorial":
        return generator.generate_combinatorial_palindrome(min_length)
    else:
        # Default to constructive
        return generator.generate_constructive_palindrome(min_length)

# For testing
if __name__ == "__main__":
    # Test the generator with different strategies
    generator = ConstructivePalindromeGenerator()
    
    print("\nTesting constructive palindrome generation:")
    palindrome = generator.generate_constructive_palindrome(min_length=80)
    normalized = normalize(palindrome)
    print(f"Palindrome: {palindrome}")
    print(f"Normalized: {normalized}")
    print(f"Is palindrome: {normalized == normalized[::-1]}")
    print(f"Length: {len(normalized)}")
    
    print("\nTesting center-out palindrome generation:")
    palindrome = generator.generate_center_out_palindrome(min_length=80)
    normalized = normalize(palindrome)
    print(f"Palindrome: {palindrome}")
    print(f"Normalized: {normalized}")
    print(f"Is palindrome: {normalized == normalized[::-1]}")
    print(f"Length: {len(normalized)}")
    
    print("\nTesting mirrored palindrome generation:")
    palindrome = generator.generate_mirrored_palindrome(min_length=80)
    normalized = normalize(palindrome)
    print(f"Palindrome: {palindrome}")
    print(f"Normalized: {normalized}")
    print(f"Is palindrome: {normalized == normalized[::-1]}")
    print(f"Length: {len(normalized)}")
    
    print("\nTesting combinatorial palindrome generation:")
    palindrome = generator.generate_combinatorial_palindrome(min_length=80)
    normalized = normalize(palindrome)
    print(f"Palindrome: {palindrome}")
    print(f"Normalized: {normalized}")
    print(f"Is palindrome: {normalized == normalized[::-1]}")
    print(f"Length: {len(normalized)}")
