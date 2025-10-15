#!/usr/bin/env python3
"""
Correct Palindrome Generator

This module generates character-level palindromes using a constructive approach.
Unlike the broken previous implementation, this uses proper middle-out construction
that guarantees valid palindromes every time.

Key principles:
1. Build from a palindromic core outward
2. Add characters symmetrically (same char to both ends)
3. Insert spaces/punctuation symmetrically to preserve palindrome property
4. Accept that truly grammatical palindromes are extremely rare
"""

import random
import re
from typing import Tuple, List


def normalize(text: str) -> str:
    """Remove all non-alphabetic characters and convert to lowercase."""
    return re.sub(r'[^a-zA-Z]', '', text).lower()


def is_palindrome(text: str) -> bool:
    """Check if text is a character-level palindrome."""
    normalized = normalize(text)
    return normalized == normalized[::-1]


class PalindromeGenerator:
    """Generate valid character-level palindromes using constructive methods."""
    
    # Palindromic words to use as cores
    PALINDROME_WORDS = [
        'a', 'i', 'aa', 'aha', 'bib', 'bob', 'bub', 'civic', 'dad', 
        'deed', 'did', 'dud', 'eke', 'eve', 'eye', 'gag', 'gig',
        'hah', 'huh', 'kayak', 'level', 'madam', 'minim', 'mom',
        'noon', 'nun', 'pap', 'peep', 'pip', 'pop', 'pup', 'radar',
        'redder', 'refer', 'repaper', 'reviver', 'rotator', 'rotor',
        'sagas', 'sees', 'selles', 'semes', 'senes', 'seres', 'sexes',
        'shahs', 'sis', 'solos', 'stats', 'stets', 'tenet', 'terret',
        'tit', 'tat', 'toot', 'tot', 'tut', 'wow'
    ]
    
    # Common short words for building
    SHORT_WORDS = ['a', 'i', 'am', 'an', 'as', 'at', 'be', 'by', 'do', 
                   'go', 'he', 'hi', 'if', 'in', 'is', 'it', 'me', 'my',
                   'no', 'of', 'on', 'or', 'so', 'to', 'up', 'us', 'we']
    
    def __init__(self, seed: int = None):
        """Initialize generator with optional random seed."""
        if seed is not None:
            random.seed(seed)
    
    def generate_from_core(self, min_length: int = 60, 
                          use_spaces: bool = True) -> Tuple[str, str]:
        """
        Generate a palindrome by starting with a palindromic core and building outward.
        
        Args:
            min_length: Minimum length of the normalized palindrome
            use_spaces: Whether to insert spaces (makes it more readable but less grammatical)
        
        Returns:
            Tuple of (readable_text, normalized_text)
        """
        # Start with a random palindromic word as the core
        core = random.choice(self.PALINDROME_WORDS)
        normalized = core
        
        # Build outward until we reach minimum length
        while len(normalized) < min_length:
            # Choose what to add
            if random.random() < 0.7:
                # Add a single letter
                letter = random.choice('abcdefghijklmnopqrstuvwxyz')
                normalized = letter + normalized + letter
            else:
                # Add a short palindromic sequence
                sequences = ['aa', 'aba', 'aca', 'ada', 'aha', 'ala', 'ama', 
                           'ana', 'apa', 'ara', 'asa', 'ata', 'ava', 'awa']
                seq = random.choice(sequences)
                normalized = seq + normalized + seq
        
        # Convert to readable text with spacing if requested
        if use_spaces:
            readable = self._add_spacing(normalized)
        else:
            readable = normalized
        
        return readable, normalized
    
    def generate_sentence_style(self, min_length: int = 60) -> Tuple[str, str]:
        """
        Generate a palindrome that looks more like sentences.
        Still not grammatically perfect, but more readable.
        
        Args:
            min_length: Minimum length of the normalized palindrome
        
        Returns:
            Tuple of (readable_text, normalized_text)
        """
        # Start with a palindromic phrase core
        cores = [
            "a man a plan a canal panama",
            "never odd or even",
            "rats live on no evil star",
            "was it a car or a cat i saw",
            "no lemon no melon",
            "step on no pets",
            "madam im adam",
            "able was i ere i saw elba"
        ]
        
        core_phrase = random.choice(cores)
        normalized = normalize(core_phrase)
        
        # Verify it's a palindrome, if not, fix it
        if normalized != normalized[::-1]:
            # Take first half and mirror it
            half = len(normalized) // 2
            if len(normalized) % 2 == 1:
                normalized = normalized[:half+1] + normalized[:half][::-1]
            else:
                normalized = normalized[:half] + normalized[:half][::-1]
        
        # Extend if needed
        while len(normalized) < min_length:
            # Add letters to both ends
            letter = random.choice('abcdefghijklmnopqrstuvwxyz')
            normalized = letter + normalized + letter
        
        # Convert back to readable format
        readable = self._add_sentence_formatting(normalized)
        
        return readable, normalized
    
    def generate_mirrored_phrase(self, min_length: int = 60) -> Tuple[str, str]:
        """
        Generate a palindrome by creating a phrase and mirroring it.
        
        Args:
            min_length: Minimum length of the normalized palindrome
        
        Returns:
            Tuple of (readable_text, normalized_text)
        """
        # Build the first half
        target_half_length = min_length // 2
        first_half = ""
        
        # Add some words to the first half
        while len(first_half) < target_half_length:
            if random.random() < 0.3 and self.PALINDROME_WORDS:
                word = random.choice(self.PALINDROME_WORDS)
            else:
                word = random.choice(self.SHORT_WORDS)
            first_half += word
        
        # Trim to exact length if over
        first_half = first_half[:target_half_length]
        
        # Mirror it to create the palindrome
        if min_length % 2 == 1:
            # Odd length - need a center character
            center = random.choice('abcdefghijklmnopqrstuvwxyz')
            normalized = first_half + center + first_half[::-1]
        else:
            # Even length - just mirror
            normalized = first_half + first_half[::-1]
        
        # Add readable formatting
        readable = self._add_spacing(normalized)
        
        return readable, normalized
    
    def _add_spacing(self, normalized: str) -> str:
        """
        Add spaces to normalized text to make it more readable.
        Spaces must be added symmetrically to preserve palindrome property.
        
        Args:
            normalized: The normalized palindrome string
        
        Returns:
            Readable text with spaces
        """
        # Find the middle point
        length = len(normalized)
        middle = length // 2
        
        # Determine where to place spaces in the first half
        # We'll try to break it into word-like chunks
        first_half = normalized[:middle]
        space_positions = set()
        
        # Add spaces every 3-5 characters
        pos = 0
        while pos < len(first_half):
            chunk_size = random.randint(2, 5)
            pos += chunk_size
            if pos < len(first_half):
                space_positions.add(pos)
        
        # Build the text with symmetric spacing
        result = []
        for i, char in enumerate(normalized):
            result.append(char)
            
            # Add space if this position should have one
            if i in space_positions:
                result.append(' ')
            
            # Mirror position: add space if mirror position has one
            mirror_pos = length - 1 - i
            if mirror_pos in space_positions and mirror_pos != i:
                result.append(' ')
        
        text = ''.join(result)
        
        # Capitalize first letter and add punctuation
        text = text.strip()
        if text:
            text = text[0].upper() + text[1:]
            # Add periods at roughly sentence-length intervals
            words = text.split()
            sentences = []
            current = []
            for i, word in enumerate(words):
                current.append(word)
                if len(current) >= 5 or i == len(words) - 1:
                    sentence = ' '.join(current)
                    if sentence:
                        sentences.append(sentence + '.')
                    current = []
            text = ' '.join(sentences)
        
        return text
    
    def _add_sentence_formatting(self, normalized: str) -> str:
        """
        Add sentence-like formatting to normalized text.
        
        Args:
            normalized: The normalized palindrome string
        
        Returns:
            Text formatted as sentences
        """
        # Try to identify recognizable words
        result = []
        i = 0
        
        # Common words to look for
        common_words = set(self.SHORT_WORDS + [w for w in self.PALINDROME_WORDS if len(w) <= 5])
        
        while i < len(normalized):
            # Try to find a word starting at position i
            found = False
            for word_len in range(min(8, len(normalized) - i), 0, -1):
                potential_word = normalized[i:i+word_len]
                if potential_word in common_words:
                    result.append(potential_word)
                    i += word_len
                    found = True
                    break
            
            if not found:
                # Just take 2-3 characters as a "word"
                word_len = min(random.randint(2, 3), len(normalized) - i)
                result.append(normalized[i:i+word_len])
                i += word_len
        
        # Join into sentences
        text = ' '.join(result)
        
        # Capitalize and add punctuation
        words = text.split()
        sentences = []
        current = []
        
        for i, word in enumerate(words):
            current.append(word)
            # Make sentences of 4-7 words
            if len(current) >= random.randint(4, 7) or i == len(words) - 1:
                if current:
                    sentence = ' '.join(current)
                    sentence = sentence[0].upper() + sentence[1:] + '.'
                    sentences.append(sentence)
                    current = []
        
        return ' '.join(sentences)
    
    def generate(self, min_length: int = 60, 
                 method: str = 'auto') -> Tuple[str, str, bool]:
        """
        Generate a palindrome using the specified method.
        
        Args:
            min_length: Minimum length of normalized palindrome
            method: Generation method ('auto', 'core', 'sentence', 'mirror')
        
        Returns:
            Tuple of (readable_text, normalized_text, is_valid)
        """
        if method == 'auto':
            # Choose a random method
            method = random.choice(['core', 'sentence', 'mirror'])
        
        if method == 'core':
            readable, normalized = self.generate_from_core(min_length)
        elif method == 'sentence':
            readable, normalized = self.generate_sentence_style(min_length)
        elif method == 'mirror':
            readable, normalized = self.generate_mirrored_phrase(min_length)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Validate
        is_valid = is_palindrome(readable)
        
        return readable, normalized, is_valid


def generate_palindrome(min_length: int = 60, 
                       method: str = 'auto',
                       seed: int = None) -> Tuple[str, str, bool]:
    """
    Convenience function to generate a palindrome.
    
    Args:
        min_length: Minimum length of the normalized palindrome
        method: Generation method ('auto', 'core', 'sentence', 'mirror')
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (readable_text, normalized_text, is_valid)
    """
    generator = PalindromeGenerator(seed=seed)
    return generator.generate(min_length=min_length, method=method)


if __name__ == "__main__":
    # Quick test
    print("Testing Palindrome Generator\n" + "="*50)
    
    for method in ['core', 'sentence', 'mirror']:
        print(f"\nMethod: {method}")
        readable, normalized, is_valid = generate_palindrome(
            min_length=60, 
            method=method
        )
        print(f"Valid: {'✅' if is_valid else '❌'}")
        print(f"Length: {len(normalized)}")
        print(f"Text: {readable}")
        print(f"Normalized: {normalized[:50]}...{normalized[-50:]}")
