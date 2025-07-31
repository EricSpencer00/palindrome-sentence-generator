#!/usr/bin/env python3
"""
Semantic Tools for Palindrome Generation

This module provides semantic coherence tools to improve the quality
of generated palindromes beyond simple grammar checks.
"""

import spacy
import logging
import numpy as np
from collections import defaultdict

# Configure logging
logging.basicConfig(
    filename='palindrome.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load spaCy model with word vectors
try:
    nlp = spacy.load("en_core_web_md")  # Use medium model with word vectors
except OSError:
    logging.warning("SpaCy medium model not found. Downloading...")
    import os
    os.system("python -m spacy download en_core_web_md")
    nlp = spacy.load("en_core_web_md")

class SemanticAnalyzer:
    """Provides tools for semantic analysis and coherence improvement."""
    
    def __init__(self):
        self.nlp = nlp
        self.common_themes = self._load_common_themes()
        
    def _load_common_themes(self):
        """
        Load common themes and their associated words for semantic
        coherence evaluation.
        """
        themes = {
            "nature": ["tree", "river", "mountain", "forest", "lake", "ocean", "sky", 
                      "sun", "moon", "star", "bird", "animal", "flower", "garden"],
            
            "technology": ["computer", "software", "internet", "digital", "device", 
                          "phone", "app", "program", "code", "data", "network"],
            
            "food": ["eat", "meal", "cook", "food", "recipe", "dish", "taste", 
                    "flavor", "kitchen", "restaurant", "dinner", "lunch"],
            
            "travel": ["journey", "trip", "vacation", "visit", "travel", "explore",
                      "destination", "map", "route", "tour", "guide", "path"],
            
            "art": ["create", "art", "music", "paint", "draw", "design", "color",
                   "artist", "creative", "image", "picture", "sculpture"],
            
            "time": ["time", "day", "year", "hour", "minute", "second", "moment",
                    "past", "present", "future", "age", "era", "history"],
            
            "emotion": ["feel", "emotion", "happy", "sad", "love", "fear", "anger",
                       "joy", "peace", "stress", "worry", "calm", "excited"]
        }
        
        # Add vector representations for each theme
        theme_vectors = {}
        for theme, words in themes.items():
            # Create an average vector for each theme
            vectors = [self.nlp(word).vector for word in words if self.nlp(word).has_vector]
            if vectors:
                theme_vectors[theme] = np.mean(vectors, axis=0)
                
        return {
            "themes": themes,
            "vectors": theme_vectors
        }
    
    def identify_themes(self, text):
        """
        Identify potential themes in the text based on word similarity.
        Returns a dictionary of themes and their relevance scores.
        """
        doc = self.nlp(text)
        content_words = [token for token in doc if not token.is_stop and token.has_vector]
        
        if not content_words:
            return {}
            
        # Calculate average similarity to each theme
        theme_scores = {}
        for theme, theme_vector in self.common_themes["vectors"].items():
            similarities = []
            for word in content_words:
                if word.has_vector:
                    similarity = np.dot(word.vector, theme_vector) / (np.linalg.norm(word.vector) * np.linalg.norm(theme_vector))
                    similarities.append(similarity)
            
            if similarities:
                theme_scores[theme] = np.mean(similarities)
        
        # Normalize scores
        if theme_scores:
            max_score = max(theme_scores.values())
            for theme in theme_scores:
                theme_scores[theme] /= max_score
                
        return theme_scores
    
    def evaluate_semantic_coherence(self, text):
        """
        Evaluate the semantic coherence of a text from 0 to 100.
        Higher scores indicate better semantic coherence.
        """
        doc = self.nlp(text)
        
        # Base score
        score = 50
        
        # Check for content words (not stop words)
        content_words = [token for token in doc if not token.is_stop and token.is_alpha]
        if len(content_words) < 3:
            return max(0, score - 30)  # Penalize very short or empty texts
        
        # Calculate average similarity between consecutive content words
        similarities = []
        prev_word = None
        for word in content_words:
            if prev_word and word.has_vector and prev_word.has_vector:
                similarity = word.similarity(prev_word)
                similarities.append(similarity)
            prev_word = word
        
        # Adjust score based on word similarities
        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
            score += avg_similarity * 30  # Boost score for related words
        
        # Check for theme coherence
        theme_scores = self.identify_themes(text)
        if theme_scores:
            dominant_theme_score = max(theme_scores.values())
            score += dominant_theme_score * 20  # Boost score for strong thematic coherence
        
        # Normalize score
        return max(0, min(score, 100))
    
    def suggest_thematic_words(self, text, max_suggestions=5):
        """
        Suggest thematically related words to enhance semantic coherence.
        Returns a list of suggested words that fit the dominant theme.
        """
        theme_scores = self.identify_themes(text)
        if not theme_scores:
            return []
            
        # Find the dominant theme
        dominant_theme = max(theme_scores.items(), key=lambda x: x[1])[0]
        theme_words = self.common_themes["themes"][dominant_theme]
        
        # Check which theme words are not already in the text
        doc = self.nlp(text.lower())
        text_words = [token.text for token in doc]
        
        suggestions = []
        for word in theme_words:
            if word not in text_words and len(suggestions) < max_suggestions:
                # Verify the word is palindrome-friendly (not too long)
                if len(word) <= 6:
                    suggestions.append(word)
        
        return suggestions
    
    def find_semantically_similar_palindromes(self, word, threshold=0.5):
        """
        Find words that are semantically similar to the input word
        and are palindromes or have palindromic potential.
        """
        word_doc = self.nlp(word)
        if not word_doc.has_vector:
            return []
            
        palindrome_words = ["level", "radar", "civic", "madam", "refer", "noon", 
                           "rotor", "kayak", "reviver", "redder", "racecar", "deed"]
        
        # Add common short words that could help build palindromes
        short_words = ["eye", "mom", "dad", "wow", "pop", "tat", "did", "sees", "peep"]
        all_words = palindrome_words + short_words
        
        similar_words = []
        for candidate in all_words:
            candidate_doc = self.nlp(candidate)
            if candidate_doc.has_vector:
                similarity = word_doc.similarity(candidate_doc)
                if similarity > threshold:
                    similar_words.append((candidate, similarity))
        
        # Sort by similarity
        similar_words.sort(key=lambda x: x[1], reverse=True)
        return [word for word, sim in similar_words]
    
    def improve_semantic_coherence(self, palindrome):
        """
        Attempt to improve the semantic coherence of a palindrome while
        maintaining its palindromic property.
        """
        original_score = self.evaluate_semantic_coherence(palindrome)
        logging.info(f"Original semantic coherence score: {original_score}/100")
        
        words = palindrome.split()
        improved = False
        best_palindrome = palindrome
        best_score = original_score
        
        # Strategy 1: Replace semantically unrelated words with thematic alternatives
        theme_scores = self.identify_themes(palindrome)
        if theme_scores:
            # Find the dominant theme
            dominant_theme = max(theme_scores.items(), key=lambda x: x[1])[0]
            
            # Identify words that could be replaced (especially common filler words)
            from grammar_validator import GrammarValidator
            grammar_validator = GrammarValidator()
            
            for i, word in enumerate(words):
                if word.lower() in ["a", "the", "and", "or", "but", "of", "in", "on", "at"]:
                    continue  # Skip common determiners and prepositions
                
                # Find a thematically related word that's palindrome-friendly
                similar_words = self.find_semantically_similar_palindromes(word)
                for replacement in similar_words:
                    if replacement != word:
                        # Try the replacement
                        test_words = words.copy()
                        test_words[i] = replacement
                        
                        # For palindromes, we need to replace the mirror word too
                        mirror_idx = len(words) - 1 - i
                        if 0 <= mirror_idx < len(words):
                            # Check if it's the exact same position (center word)
                            if mirror_idx == i:
                                # For center word, just replace once
                                pass
                            else:
                                # Replace mirror word with reversed replacement
                                test_words[mirror_idx] = replacement[::-1] if not self._is_palindrome(replacement) else replacement
                        
                        test_palindrome = " ".join(test_words)
                        if grammar_validator.is_palindrome(test_palindrome):
                            test_score = self.evaluate_semantic_coherence(test_palindrome)
                            if test_score > best_score:
                                best_palindrome = test_palindrome
                                best_score = test_score
                                improved = True
                                logging.info(f"Improved semantic coherence: {original_score} -> {test_score} (replacing {word} with {replacement})")
                                break
        
        # Strategy 2: Insert thematic connectors (for texts with poor coherence)
        if best_score < 60:
            from grammar_validator import GrammarValidator
            grammar_validator = GrammarValidator()
            
            # Try to insert thematic words at strategic positions
            theme_words = self.suggest_thematic_words(palindrome)
            if theme_words:
                for theme_word in theme_words:
                    # Try inserting the word at different positions
                    for pos in [1, len(words) // 4, len(words) // 2]:
                        if pos < len(words):
                            test_words = words.copy()
                            
                            # Insert the word
                            test_words.insert(pos, theme_word)
                            
                            # Insert mirror word at the appropriate position
                            mirror_pos = len(test_words) - pos
                            if 0 <= mirror_pos < len(test_words) + 1:
                                mirror_word = theme_word[::-1] if not self._is_palindrome(theme_word) else theme_word
                                test_words.insert(mirror_pos, mirror_word)
                                
                            test_palindrome = " ".join(test_words)
                            if grammar_validator.is_palindrome(test_palindrome):
                                test_score = self.evaluate_semantic_coherence(test_palindrome)
                                if test_score > best_score:
                                    best_palindrome = test_palindrome
                                    best_score = test_score
                                    improved = True
                                    logging.info(f"Improved semantic coherence: {original_score} -> {test_score} (inserting {theme_word})")
                                    break
        
        if improved and best_score > original_score:
            return best_palindrome, best_score
        else:
            return palindrome, original_score
    
    def _is_palindrome(self, text):
        """Check if a string is a palindrome (ignoring spaces and punctuation)."""
        # Remove all non-alphanumeric characters and convert to lowercase
        processed_text = ''.join(c.lower() for c in text if c.isalnum())
        # Check if the processed text reads the same forward and backward
        return processed_text == processed_text[::-1]

def main():
    """Test the semantic analyzer with some sample palindromes."""
    analyzer = SemanticAnalyzer()
    
    test_palindromes = [
        "A man a plan a canal Panama",
        "Rats live on no evil star",
        "Step on no pets",
        "a a a a a a a a a radar a a a a a a a a a",
        "Mom wow mom eye sees mom wow mom",
        "I I I a lunula a a man a plan a mom never odd or even never odd or even never odd or even A Doc, note: I dissent. A fast never prevents a fatness. I diet on cod a neve ro ddo reven neve ro ddo reven neve ro ddo reven mom a nalp a nam a a a lunula I I I"
    ]
    
    for palindrome in test_palindromes:
        print(f"\nAnalyzing: '{palindrome[:50]}...' if longer than 50 chars")
        
        # Score semantic coherence
        score = analyzer.evaluate_semantic_coherence(palindrome)
        print(f"Semantic coherence score: {score}/100")
        
        # Identify themes
        themes = analyzer.identify_themes(palindrome)
        if themes:
            print("Detected themes:")
            for theme, relevance in sorted(themes.items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"  - {theme}: {relevance:.2f}")
        
        # Suggest related words
        suggestions = analyzer.suggest_thematic_words(palindrome)
        if suggestions:
            print("Suggested thematic words:", ", ".join(suggestions))
        
        # Try to improve
        improved, improved_score = analyzer.improve_semantic_coherence(palindrome)
        if improved != palindrome:
            print(f"Improved version: '{improved[:50]}...' if longer than 50 chars")
            print(f"Improved score: {improved_score}/100")
        else:
            print("No semantic improvements found")
        
        print("-" * 50)

if __name__ == "__main__":
    main()
