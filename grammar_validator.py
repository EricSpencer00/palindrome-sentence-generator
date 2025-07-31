#!/usr/bin/env python3
"""
Grammar Validator for Palindromes

This script provides functions to validate and improve the grammatical 
structure of palindrome sentences.
"""

import spacy
import logging
import re
import random
from nltk.tokenize import word_tokenize
from collections import defaultdict

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logging.error("spaCy model not found. Please run: python -m spacy download en_core_web_sm")
    raise

class GrammarValidator:
    def __init__(self):
        self.nlp = nlp
        self.grammar_patterns = self._define_grammar_patterns()
        
    def _define_grammar_patterns(self):
        """Define common grammatical patterns for short sentences."""
        return [
            # Simple patterns
            ["DET", "NOUN"],                       # "The cat"
            ["DET", "ADJ", "NOUN"],                # "The big cat"
            ["PRON", "VERB"],                      # "I see"
            ["PRON", "VERB", "DET", "NOUN"],       # "I see the cat"
            
            # More complex patterns
            ["DET", "NOUN", "VERB"],               # "The cat sleeps"
            ["DET", "NOUN", "VERB", "DET", "NOUN"] # "The cat sees the dog"
        ]
    
    def analyze_grammar(self, text):
        """Analyze the grammatical structure of a text."""
        doc = self.nlp(text)
        
        # Extract parts of speech
        pos_tags = [(token.text, token.pos_) for token in doc]
        
        # Check sentence boundaries
        sentences = list(doc.sents)
        
        return {
            "pos_tags": pos_tags,
            "sentence_count": len(sentences),
            "tokens": [token.text for token in doc],
            "tokens_count": len(doc)
        }
    
    def score_grammatical_quality(self, text):
        """
        Score the grammatical quality of a text from 0 to 100.
        Higher scores indicate better grammatical structure.
        """
        if not text or len(text.strip()) < 3:
            return 0
            
        doc = self.nlp(text)
        
        # Base score
        score = 50
        
        # Check for sentence completeness
        for sent in doc.sents:
            # Does the sentence have a subject?
            has_subject = any(token.dep_ in ('nsubj', 'nsubjpass') for token in sent)
            # Does the sentence have a verb?
            has_verb = any(token.pos_ == 'VERB' for token in sent)
            
            if has_subject and has_verb:
                score += 10  # Complete sentence with subject and verb
            elif has_subject or has_verb:
                score += 5   # At least has a subject or verb
            
            # Penalize very short sentences
            if len(sent) < 3:
                score -= 5
        
        # Check for determiners before nouns
        for i, token in enumerate(doc):
            if token.pos_ == 'NOUN' and i > 0:
                prev_token = doc[i-1]
                if prev_token.pos_ in ('DET', 'ADJ'):
                    score += 2  # Proper noun phrases
        
        # Check for pattern matches
        pos_sequence = [token.pos_ for token in doc]
        for pattern in self.grammar_patterns:
            # Check if any subsequence matches the pattern
            pattern_len = len(pattern)
            for i in range(len(pos_sequence) - pattern_len + 1):
                if pos_sequence[i:i+pattern_len] == pattern:
                    score += 5  # Found a valid grammar pattern
                    break
        
        # Normalize score
        score = max(0, min(score, 100))
        
        return score
    
    def suggest_grammar_improvements(self, text):
        """Suggest improvements to the grammatical structure of the text."""
        doc = self.nlp(text)
        suggestions = []
        
        # Check for missing determiners
        for token in doc:
            if token.pos_ == 'NOUN' and token.dep_ not in ('compound', 'fixed'):
                # Check if there's a determiner
                has_det = any(child.pos_ == 'DET' for child in token.children)
                if not has_det and not token.text[0].isupper():  # Check if not proper noun
                    suggestions.append(f"Consider adding a determiner before '{token.text}'")
        
        # Check for subject-verb agreement
        for token in doc:
            if token.pos_ == 'VERB':
                subj = [child for child in token.children if child.dep_ in ('nsubj', 'nsubjpass')]
                if subj and subj[0].pos_ == 'PRON':
                    # Simple check - not comprehensive
                    if subj[0].text.lower() in ('i', 'you', 'we', 'they') and token.text.endswith('s'):
                        suggestions.append(f"Verb '{token.text}' may not agree with subject '{subj[0].text}'")
        
        # Check for repeated words (common in palindromes but can be excessive)
        word_counts = defaultdict(int)
        for token in doc:
            if token.is_alpha and not token.is_stop:
                word_counts[token.text.lower()] += 1
        
        repetitive_words = [word for word, count in word_counts.items() if count > 3]
        if repetitive_words:
            suggestions.append(f"Excessive repetition of: {', '.join(repetitive_words)}")
            
        # Check for sentence coherence
        if len(list(doc.sents)) > 1:
            suggestions.append("Consider improving the connection between sentences for better flow")
            
        # Check for lack of structure
        if len(doc) > 10 and not any(token.pos_ == 'VERB' for token in doc):
            suggestions.append("Text lacks verbs, consider adding structure with action words")
            
        # Check for grammatical patterns
        has_structure = False
        pos_sequence = [token.pos_ for token in doc]
        for pattern in self.grammar_patterns:
            pattern_len = len(pattern)
            for i in range(len(pos_sequence) - pattern_len + 1):
                if pos_sequence[i:i+pattern_len] == pattern:
                    has_structure = True
                    break
            if has_structure:
                break
                
        if not has_structure and len(doc) > 5:
            suggestions.append("Text lacks common grammatical patterns")
            
        # Limit suggestions to a reasonable number
        return suggestions[:5]
    
    def improve_palindrome_grammar(self, palindrome):
        """
        Attempt to improve the grammar of a palindrome while maintaining its palindromic property.
        Returns the improved palindrome or the original if no improvements could be made.
        """
        # First, analyze the current grammatical quality
        original_score = self.score_grammatical_quality(palindrome)
        
        # If already good grammatically, return as is
        if original_score > 80:
            return palindrome, original_score
            
        # Try to make improvements by breaking into chunks and restructuring
        words = word_tokenize(palindrome)
        improved = False
        best_palindrome = palindrome
        best_score = original_score
        
        # Strategy 1: Add determiners where missing before nouns (if palindrome property can be maintained)
        for i in range(len(words)):
            if i >= len(words):  # Guard against index changes
                break
                
            doc = self.nlp(words[i])
            if len(doc) == 1 and doc[0].pos_ == 'NOUN' and i > 0 and len(words) > i + 1:
                # Check if we can add determiners like "a" or "the" before and after
                for det in ['a', 'the', 'my', 'your']:
                    test_words = words.copy()
                    test_words = test_words[:i] + [det] + [test_words[i]] + [self._reverse_word(det)] + test_words[i+1:]
                    test_palindrome = ' '.join(test_words)
                    
                    if self._is_palindrome(test_palindrome):
                        test_score = self.score_grammatical_quality(test_palindrome)
                        if test_score > best_score:
                            best_palindrome = test_palindrome
                            best_score = test_score
                            improved = True
                            words = test_words  # Update for further improvements
                
        # Strategy 2: Break long phrases into smaller chunks with better structure
        if len(words) > 7:
            # Try to insert punctuation to create natural breaks
            mid = len(words) // 2
            for punct in [',', ';', ':']:
                if mid > 0 and mid < len(words) - 1:
                    test_words = words.copy()
                    test_words = test_words[:mid] + [punct] + test_words[mid:]
                    test_palindrome = ' '.join(test_words)
                    
                    if self._is_palindrome(test_palindrome):
                        test_score = self.score_grammatical_quality(test_palindrome)
                        if test_score > best_score:
                            best_palindrome = test_palindrome
                            best_score = test_score
                            improved = True
                            words = test_words
            
            # Try natural conjunctions if they maintain palindrome property
            for i in range(2, len(words) - 2):
                for conj in ['and', 'or', 'but']:
                    rev_conj = self._reverse_word(conj)
                    test_words = words.copy()
                    test_words = test_words[:i] + [conj] + test_words[i:]
                    test_palindrome = ' '.join(test_words)
                    
                    if self._is_palindrome(test_palindrome) and rev_conj in ' '.join(test_words[i:]):
                        test_score = self.score_grammatical_quality(test_palindrome)
                        if test_score > best_score:
                            best_palindrome = test_palindrome
                            best_score = test_score
                            improved = True
                            words = test_words
        
        # Strategy 3: Replace repetitive words with more meaningful palindromic words
        a_count = words.count('a')
        if a_count > 5:
            # Find palindrome words that could replace 'a'
            palindrome_words = ['I', 'eye', 'mom', 'dad', 'wow', 'noon', 'level', 'deed']
            
            # Replace some 'a's with other palindrome words
            a_indices = [i for i, word in enumerate(words) if word == 'a']
            if len(a_indices) >= 4:  # Only if we have enough 'a's to work with
                replace_indices = random.sample(a_indices, min(4, len(a_indices)))
                for idx in replace_indices:
                    for replacement in palindrome_words:
                        test_words = words.copy()
                        test_words[idx] = replacement
                        
                        # Find the matching 'a' on the other side of the palindrome
                        mirror_idx = len(words) - 1 - idx
                        if 0 <= mirror_idx < len(words) and words[mirror_idx] == 'a':
                            test_words[mirror_idx] = replacement
                            
                        test_palindrome = ' '.join(test_words)
                        if self._is_palindrome(test_palindrome):
                            test_score = self.score_grammatical_quality(test_palindrome)
                            if test_score > best_score:
                                best_palindrome = test_palindrome
                                best_score = test_score
                                improved = True
                                words = test_words
        
        # Strategy 4: Add subject-verb structure if missing
        has_verb = any(self.nlp(word)[0].pos_ == 'VERB' for word in words)
        has_subject = any(self.nlp(word)[0].pos_ == 'PRON' for word in words)
        
        if not has_verb or not has_subject:
            # Try to add a simple subject-verb structure
            for subject in ['I', 'we', 'you']:
                for verb in ['see', 'love', 'need']:
                    test_words = words.copy()
                    
                    # Try inserting at different positions
                    for pos in [1, len(words) // 4, len(words) // 2]:
                        if pos < len(test_words):
                            test_with_subj_verb = test_words[:pos] + [subject, verb] + test_words[pos:]
                            test_palindrome = ' '.join(test_with_subj_verb)
                            
                            if self._is_palindrome(test_palindrome):
                                test_score = self.score_grammatical_quality(test_palindrome)
                                if test_score > best_score:
                                    best_palindrome = test_palindrome
                                    best_score = test_score
                                    improved = True
        
        # Strategy 5: Add grammatical structure near the beginning and end
        if len(words) > 10:
            # Try common sentence starters
            for starter in ['the', 'a', 'this', 'my', 'in the']:
                test_words = [starter] + words + [self._reverse_word(starter)]
                test_palindrome = ' '.join(test_words)
                
                if self._is_palindrome(test_palindrome):
                    test_score = self.score_grammatical_quality(test_palindrome)
                    if test_score > best_score:
                        best_palindrome = test_palindrome
                        best_score = test_score
                        improved = True
        
        if improved and best_score > original_score and self._is_palindrome(best_palindrome):
            return best_palindrome, best_score
        else:
            return palindrome, original_score
    
    def _is_palindrome(self, text):
        """Check if a string is a palindrome (ignoring spaces and punctuation)."""
        # Remove all non-alphanumeric characters and convert to lowercase
        processed_text = ''.join(c.lower() for c in text if c.isalnum())
        # Check if the processed text reads the same forward and backward
        return processed_text == processed_text[::-1]
    
    def _reverse_word(self, word):
        """Return the reverse of a word."""
        return word[::-1]

def main():
    """Test the grammar validator with some example palindromes."""
    validator = GrammarValidator()
    
    test_palindromes = [
        "A man a plan a canal Panama",
        "Was it a car or a cat I saw",
        "Madam I'm Adam",
        "a a a a a a a a a radar a a a a a a a a a",
        "Mom wow mom eye sees mom wow mom",
        "Step on no pets kayak step on no pets"
    ]
    
    print("Analyzing palindrome grammar quality:\n")
    
    for palindrome in test_palindromes:
        score = validator.score_grammatical_quality(palindrome)
        suggestions = validator.suggest_grammar_improvements(palindrome)
        improved, improved_score = validator.improve_palindrome_grammar(palindrome)
        
        print(f"Original: \"{palindrome}\"")
        print(f"Grammar score: {score}/100")
        if suggestions:
            print("Suggestions:")
            for suggestion in suggestions:
                print(f"  - {suggestion}")
                
        if improved != palindrome:
            print(f"Improved: \"{improved}\"")
            print(f"Improved score: {improved_score}/100")
            
        print("-" * 80)

if __name__ == "__main__":
    main()
