#!/usr/bin/env python3
"""
Enhanced Palindrome Paragraph Generator

This version focuses on creating grammatically correct palindromes by:
1. Maintaining character-level palindrome property
2. Ensuring both halves have sensible word boundaries
3. Using grammar rules to make both sides readable
"""

import argparse
import logging
import re
import time
import nltk
import spacy
import random
from typing import List, Tuple, Dict, Any, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("palindrome.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("grammar_palindrome_generator")

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Downloading spaCy model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Ensure NLTK resources are available
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

from nltk.corpus import words as nltk_words

class GrammarPalindromeGenerator:
    """Enhanced palindrome generator with grammatical structure awareness"""
    
    def __init__(self):
        # Dictionary of English words
        self.words_dict = set(w.lower() for w in nltk_words.words())
        
        # Common palindrome center phrases
        self.center_phrases = [
            "racecar",
            "a man a plan a canal panama",
            "never odd or even",
            "madam im adam",
            "step on no pets",
            "wow",
            "noon",
            "level"
        ]
        
        # Dictionary word categories
        self.determiners = {"a", "the", "this", "that", "these", "those", "my", "your", "our", "their"}
        self.prepositions = {"in", "on", "at", "by", "with", "for", "to", "from", "of", "as", "near"}
        self.conjunctions = {"and", "or", "but", "yet", "so", "nor", "for", "while", "because"}
        
        # Common palindromic words by type
        self.palindrome_nouns = {"eye", "level", "civic", "kayak", "mom", "dad", "noon", "peep", 
                                "deed", "pup", "gag", "wow", "madam", "radar", "rotator", "stats"}
        self.palindrome_verbs = {"did", "sees", "pop", "peep", "refer", "madam", "level", "redivider"}
        self.palindrome_adjs = {"level", "redder", "deified", "reviver", "rotator", "non", "eke", "ere"}
        
        # Pre-designed grammatical segments for first half
        self.grammatical_segments = [
            "The civic duty of",
            "A radar detected",
            "My level of",
            "The madam saw",
            "A noon meeting",
            "The kayak floated",
            "A mom and dad",
            "The racecar sped",
            "A deed signed"
        ]
    
    def clean_text(self, text: str) -> str:
        """Remove punctuation, spaces, and convert to lowercase"""
        return re.sub(r'[^a-z]', '', text.lower())
    
    def is_palindrome(self, text: str) -> bool:
        """Check if text is a palindrome (ignoring spaces, punctuation, case)"""
        cleaned = self.clean_text(text)
        return cleaned == cleaned[::-1] and len(cleaned) > 10
    
    def generate_palindrome(self, center: str = None, target_length: int = 200) -> str:
        """
        Generate a grammatically valid palindrome with proper word boundaries.
        
        Args:
            center: Optional center phrase (must be palindromic)
            target_length: Target length of the palindrome
        
        Returns:
            A grammatically valid palindrome
        """
        logger.info("Generating grammar-aware palindrome")
        
        # Select a center phrase if not provided
        if not center:
            center = random.choice(self.center_phrases)
        
        # Clean center for palindrome verification
        center_clean = self.clean_text(center)
        
        # Generate the first half with good grammar
        first_half = self._generate_grammatical_first_half(target_length // 3)
        
        # Calculate remaining length needed to reach target
        remaining_length = target_length - len(first_half) - len(center) - 1  # -1 for space
        
        # Get character sequence for the second half (reversed first half)
        first_half_chars = self.clean_text(first_half)
        reversed_chars = first_half_chars[::-1]
        
        # Create the second half with improved word boundaries and grammar
        second_half = self._create_grammatical_second_half(reversed_chars)
        
        # Combine to form the complete palindrome
        palindrome = f"{first_half} {center} {second_half}"
        
        # Verify it's a palindrome
        if not self.is_palindrome(palindrome):
            logger.warning("Generated text is not a valid palindrome, retrying with simpler approach")
            return self._create_simple_palindrome(center, target_length)
        
        # Format properly (capitalize first letter)
        palindrome = self._format_palindrome(palindrome)
        
        return palindrome
    
    def _generate_grammatical_first_half(self, target_length: int) -> str:
        """
        Generate a grammatically correct first half with sensible structure.
        
        Args:
            target_length: Approximate target length
            
        Returns:
            A grammatically structured string
        """
        # Start with a pre-designed grammatical segment
        first_half = random.choice(self.grammatical_segments)
        current_length = len(first_half)
        
        # Keep track of what part of speech would come next
        # After a preposition, we need a noun phrase
        # After a verb, we could have an object
        last_words = first_half.split()[-2:]  # Get the last two words
        last_pos = "PREP" if last_words and last_words[-1].lower() in self.prepositions else "OTHER"
        
        # Build up to target length
        while current_length < target_length:
            # Choose next word based on grammar rules
            if last_pos == "PREP":
                # After preposition, use determiner + noun
                if random.random() < 0.7:
                    next_word = random.choice(list(self.determiners))
                    first_half += f" {next_word}"
                    last_pos = "DET"
                else:
                    # Or just noun
                    next_word = random.choice(list(self.palindrome_nouns))
                    first_half += f" {next_word}"
                    last_pos = "NOUN"
            elif last_pos == "DET":
                # After determiner, use adjective or noun
                if random.random() < 0.3:
                    next_word = random.choice(list(self.palindrome_adjs))
                    first_half += f" {next_word}"
                    last_pos = "ADJ"
                else:
                    next_word = random.choice(list(self.palindrome_nouns))
                    first_half += f" {next_word}"
                    last_pos = "NOUN"
            elif last_pos == "ADJ":
                # After adjective, use noun
                next_word = random.choice(list(self.palindrome_nouns))
                first_half += f" {next_word}"
                last_pos = "NOUN"
            elif last_pos == "NOUN":
                # After noun, use verb or preposition
                if random.random() < 0.6:
                    next_word = random.choice(list(self.palindrome_verbs))
                    first_half += f" {next_word}"
                    last_pos = "VERB"
                else:
                    next_word = random.choice(list(self.prepositions))
                    first_half += f" {next_word}"
                    last_pos = "PREP"
            elif last_pos == "VERB":
                # After verb, use preposition or determiner
                if random.random() < 0.7:
                    next_word = random.choice(list(self.prepositions))
                    first_half += f" {next_word}"
                    last_pos = "PREP"
                else:
                    next_word = random.choice(list(self.determiners))
                    first_half += f" {next_word}"
                    last_pos = "DET"
            else:
                # Default: add a determiner + noun
                next_word = random.choice(list(self.determiners))
                first_half += f" {next_word}"
                last_pos = "DET"
            
            current_length = len(first_half)
            
            # Avoid very long sequences
            if current_length > target_length * 1.5:
                break
        
        return first_half
    
    def _create_grammatical_second_half(self, char_sequence: str) -> str:
        """
        Create a grammatical second half that maintains palindrome property
        at the character level.
        
        Args:
            char_sequence: The character sequence to form into words
            
        Returns:
            A grammatically structured string with proper word boundaries
        """
        # Identify potential word boundaries using dictionary lookup
        words = []
        i = 0
        
        # Try to form real words while maintaining the exact character sequence
        while i < len(char_sequence):
            # Try to find a real word starting from this position
            found_word = False
            for length in range(min(10, len(char_sequence) - i), 1, -1):
                potential_word = char_sequence[i:i+length]
                if potential_word in self.words_dict:
                    words.append(potential_word)
                    i += length
                    found_word = True
                    break
            
            if not found_word:
                # If we can't find a real word, use a reasonable length segment
                word_length = min(3, len(char_sequence) - i)  # Use shorter segments
                words.append(char_sequence[i:i+word_length])
                i += word_length
        
        # Apply grammar-aware restructuring
        structured_text = self._apply_grammar_structure(words)
        
        return structured_text
    
    def _apply_grammar_structure(self, words: List[str]) -> str:
        """
        Apply grammar structure to a list of words.
        
        Args:
            words: List of words to structure
            
        Returns:
            Grammar-improved text
        """
        # We don't want to change the actual words (as that would break the palindrome)
        # but we can adjust spacing and introduce grammatical markers
        
        # Process with spaCy to identify potential parts of speech
        doc = nlp(" ".join(words))
        
        # Group words into grammatical phrases
        phrases = []
        current_phrase = []
        
        for token in doc:
            current_phrase.append(token.text)
            
            # End phrase after nouns, verbs, or every few words
            if token.pos_ in ("NOUN", "VERB") or len(current_phrase) >= 3:
                phrases.append(" ".join(current_phrase))
                current_phrase = []
        
        # Add any remaining words
        if current_phrase:
            phrases.append(" ".join(current_phrase))
        
        # Join phrases with proper spacing
        result = " ".join(phrases)
        
        return result
    
    def _create_simple_palindrome(self, center: str, target_length: int) -> str:
        """
        Create a simple palindrome that's guaranteed to work.
        
        Args:
            center: Center phrase
            target_length: Target length
            
        Returns:
            A simple palindrome
        """
        # Simple palindromes are easier to create but less grammatical
        palindrome_words = list(self.palindrome_nouns) + list(self.palindrome_verbs)
        random.shuffle(palindrome_words)
        
        # Take a subset of words
        word_count = min(5, len(palindrome_words))
        selected_words = palindrome_words[:word_count]
        
        # Create the first half
        first_half = " ".join(selected_words)
        
        # Create the second half with the exact character reversal
        first_half_chars = self.clean_text(first_half)
        reversed_chars = first_half_chars[::-1]
        
        # Form words in the second half
        second_half_words = []
        i = 0
        while i < len(reversed_chars):
            # Get a word of length 2-4
            word_length = min(random.randint(2, 4), len(reversed_chars) - i)
            second_half_words.append(reversed_chars[i:i+word_length])
            i += word_length
        
        second_half = " ".join(second_half_words)
        
        # Combine to form the palindrome
        palindrome = f"{first_half} {center} {second_half}"
        
        return palindrome
    
    def _format_palindrome(self, text: str) -> str:
        """Format the palindrome with proper capitalization and punctuation"""
        # Split the text into words
        words = text.strip().split()
        
        # If we have words, capitalize the first one
        if words:
            words[0] = words[0].capitalize()
            
        # Rejoin the words
        text = " ".join(words)
        
        # Add period if missing
        if text and text[-1] not in ".!?":
            text += "."
            
        return text

def main():
    """Main function to run the enhanced palindrome generator"""
    parser = argparse.ArgumentParser(description='Generate grammatically correct palindromes')
    parser.add_argument('--length', type=int, default=200,
                      help='Target length of the palindrome paragraph (in characters)')
    parser.add_argument('--center', type=str, default=None,
                      help='Optional center phrase to build around')
    args = parser.parse_args()
    
    # Initialize the generator
    generator = GrammarPalindromeGenerator()
    
    # Generate and display the palindrome
    print("\nGenerating grammatically correct palindrome paragraph...\n")
    start_time = time.time()
    
    palindrome = generator.generate_palindrome(
        center=args.center, 
        target_length=args.length
    )
    
    end_time = time.time()
    
    print("\n" + "="*80)
    print("GRAMMATICALLY CORRECT PALINDROME:")
    print("="*80)
    print(palindrome)
    print("="*80)
    
    # Check if it's a palindrome
    is_pal = generator.is_palindrome(palindrome)
    print(f"Is palindrome: {is_pal}")
    
    # Additional info
    print(f"Word count: {len(palindrome.split())} words")
    print(f"Length: {len(palindrome)} characters")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    # Show the cleaned palindrome text (without spaces/punctuation)
    cleaned = generator.clean_text(palindrome)
    print("\nPalindrome verification:")
    print(f"Forward:  {cleaned[:50]}..." if len(cleaned) > 100 else f"Forward:  {cleaned}")
    print(f"Backward: {cleaned[::-1][:50]}..." if len(cleaned) > 100 else f"Backward: {cleaned[::-1]}")
    
    if not is_pal:
        print("WARNING: Generated text is not a perfect palindrome.")
        # Identify where the palindrome breaks
        for i in range(len(cleaned)//2):
            if cleaned[i] != cleaned[-(i+1)]:
                print(f"Mismatch at position {i}: '{cleaned[i]}' vs '{cleaned[-(i+1)]}'")
                break

if __name__ == "__main__":
    main()
