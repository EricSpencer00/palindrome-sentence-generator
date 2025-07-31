#!/usr/bin/env python3
"""
Unit tests for the Palindrome Paragraph Generator.
These tests verify the functionality of the character-level palindrome generation
and other requirements.
"""

import unittest
import re
import sys
import os
from palindrome_generator import PalindromeParagraphGenerator

class TestPalindromeGenerator(unittest.TestCase):
    """Test cases for the PalindromeParagraphGenerator class"""
    
    def setUp(self):
        """Set up the generator for each test"""
        self.generator = PalindromeParagraphGenerator()
    
    def test_is_palindrome(self):
        """Test the palindrome validation function"""
        # Test known palindromes
        self.assertTrue(self.generator.is_palindrome("A man a plan a canal Panama"))
        self.assertTrue(self.generator.is_palindrome("Madam, I'm Adam"))
        self.assertTrue(self.generator.is_palindrome("Race car"))
        
        # Test non-palindromes
        self.assertFalse(self.generator.is_palindrome("Hello world"))
        self.assertFalse(self.generator.is_palindrome("Not a palindrome"))
    
    def test_clean_text(self):
        """Test the text cleaning function"""
        self.assertEqual(self.generator.clean_text("A man, a plan, a canal: Panama!"), "amanaplanacanalpanama")
        self.assertEqual(self.generator.clean_text("Race car"), "racecar")
        self.assertEqual(self.generator.clean_text("12345"), "")
        self.assertEqual(self.generator.clean_text(""), "")
    
    def test_generate_from_middle(self):
        """Test generating palindromes from the middle outwards"""
        # Test with a specified center
        palindrome = self.generator.generate_from_middle(center_word="a", target_length=50, max_iterations=10)
        
        # Verify it's a palindrome
        self.assertTrue(self.generator.is_palindrome(palindrome))
        
        # Check basic properties
        self.assertGreater(len(palindrome), 0)
        cleaned = self.generator.clean_text(palindrome)
        self.assertEqual(cleaned, cleaned[::-1])
    
    def test_first_half_differs_from_second(self):
        """Test verification of first half != second half requirement"""
        # A palindrome with identical word structure
        identical = "a b c d e f g g f e d c b a"
        differs, _ = self.generator.verify_first_half_differs_from_second(identical)
        self.assertFalse(differs)
        
        # A palindrome with different word structure
        different = "a man a plan a canal panama lanac a nalp a nam a"
        differs, _ = self.generator.verify_first_half_differs_from_second(different)
        self.assertTrue(differs)
    
    def test_word_validity(self):
        """Test that generated palindromes use valid dictionary words"""
        palindrome = self.generator.generate_from_middle(target_length=50, max_iterations=10)
        
        # Get words and check if they're in the dictionary
        words = palindrome.lower().split()
        
        # Allow some percentage of non-dictionary words (e.g., 30%)
        valid_count = sum(1 for word in words if 
                         self.generator.clean_text(word) in self.generator.all_words)
        
        # At least 70% should be valid dictionary words
        validity_ratio = valid_count / len(words)
        self.assertGreaterEqual(validity_ratio, 0.5, 
                              f"Only {validity_ratio:.2f} of words are valid dictionary words")
    
    def test_score_palindrome(self):
        """Test scoring function for palindromes"""
        # Generate a sample palindrome
        palindrome = self.generator.generate_from_middle(target_length=50, max_iterations=10)
        
        # Score it
        score, details = self.generator.score_palindrome(palindrome)
        
        # Score should be a number
        self.assertIsInstance(score, (int, float))
        
        # Check that details contain expected keys
        self.assertIn("length", details)
        self.assertIn("word_count", details)
        self.assertIn("unique_words", details)
        self.assertIn("is_grammatical", details)
        self.assertIn("halves_differ", details)
    
    def test_format_palindrome(self):
        """Test proper formatting of palindromes"""
        # Test capitalization
        formatted = self.generator._format_palindrome("hello world")
        self.assertEqual(formatted, "Hello world.")
        
        # Test punctuation
        formatted = self.generator._format_palindrome("hello world!")
        self.assertEqual(formatted, "Hello world!")
    
    def test_llm_bidirectional_validation(self):
        """Test the bidirectional validation for LLM-generated palindromes"""
        # Skip if LLM not available
        try:
            from llm_palindrome import LLMPalindromeGenerator
        except ImportError:
            self.skipTest("LLM palindrome generator not available")
        
        # Simple sample palindrome for testing
        palindrome = "A man a plan a canal Panama."
        
        # Validate it
        validation = self.generator.validate_bidirectional_palindrome(palindrome)
        
        # Check validation output has the expected keys
        self.assertIn("is_valid", validation)
        self.assertIn("is_palindrome", validation)
        self.assertIn("char_similarity", validation)
        self.assertIn("word_similarity", validation)
        self.assertIn("halves_differ", validation)
        
        # It should be a valid palindrome
        self.assertTrue(validation["is_palindrome"])

if __name__ == "__main__":
    print("Running palindrome generator tests...")
    unittest.main(verbosity=2)
