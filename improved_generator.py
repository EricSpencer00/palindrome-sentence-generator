#!/usr/bin/env python3
"""
Improved Palindrome Sentence Generator

This script combines multiple generation strategies and attempts to produce
higher-quality palindromes with better grammatical structure and semantic coherence.
It includes options for parallel generation, multiple improvement attempts, and
enhanced evaluation metrics.
"""

import argparse
import logging
import time
import concurrent.futures
import random
import os
import re
import nltk
from validator import normalize, is_full_palindrome
from openai import OpenAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    # Check if we're using OpenRouter
    if api_key.startswith('sk-or'):
        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    else:
        client = OpenAI(api_key=api_key)
else:
    client = None
    logging.warning("OPENAI_API_KEY not set. LLM-based generation will not be available.")

# Download required NLTK resources if not already available
try:
    nltk.data.find('corpora/words')
    nltk.data.find('corpora/wordnet')
except LookupError:
    logging.info("Downloading required NLTK resources...")
    nltk.download('words')
    nltk.download('wordnet')

class ImprovedPalindromeGenerator:
    def __init__(self):
        self.max_attempts = 5
        self.temperature = 0.7
        
    def is_palindrome(self, text):
        """Check if a string is a palindrome (ignoring spaces and punctuation)."""
        return is_full_palindrome(text)
    
    def generate_with_multiple_attempts(self, target_length=250, attempts=10, verbose=False, use_semantic=False):
        """
        Generate palindromes with multiple attempts and select the best one based on
        grammar score, semantic coherence, and length proximity to target.
        """
        best_palindrome = None
        best_score = 0
        best_length_diff = float('inf')
        
        if verbose:
            print(f"Making {attempts} generation attempts...")
            
        for i in range(attempts):
            if verbose:
                print(f"Generation attempt {i+1}...")
                
            # Generate a palindrome using grammar-based method
            current = self.grammar_generator.generate_palindrome_paragraph(target_length)
            
            # Validate it's a palindrome
            if not self.is_palindrome(current):
                logging.warning(f"Generated text is not a palindrome: {current}")
                continue
                
            # Score grammatical quality
            grammar_score = self.validator.score_grammatical_quality(current)
            length_diff = abs(len(current) - target_length)
            
            # Score semantic quality if requested and available
            semantic_score = 0
            if use_semantic and self.semantic_analyzer:
                semantic_score = self.semantic_analyzer.evaluate_semantic_coherence(current)
                if verbose:
                    print(f"  Semantic score: {semantic_score}/100")
            
            # Apply weighted scoring - balancing grammar quality, semantic coherence, and length match
            if use_semantic and self.semantic_analyzer:
                # With semantics: 50% grammar, 30% semantics, 20% length match
                combined_score = (grammar_score * 0.5) + (semantic_score * 0.3) - (length_diff / target_length) * 20
            else:
                # Without semantics: 70% grammar, 30% length match
                combined_score = grammar_score * 0.7 - (length_diff / target_length) * 30
            
            if verbose:
                print(f"  Length: {len(current)}, Grammar score: {grammar_score}/100, Combined score: {combined_score:.1f}")
            
            # Keep track of the best palindrome
            if best_palindrome is None or combined_score > best_score:
                best_palindrome = current
                best_score = combined_score
                best_length_diff = length_diff
                
            # If we've found a very good palindrome, we can stop early
            if grammar_score > 80 and (not use_semantic or semantic_score > 70) and length_diff < target_length * 0.1:
                if verbose:
                    print("  Found excellent palindrome, stopping early.")
                break
                
        return best_palindrome
    
    def improve_grammar_aggressively(self, palindrome, max_attempts=10, verbose=False):
        """
        Apply grammar improvements more aggressively with multiple iterations.
        """
        initial_score = self.validator.score_grammatical_quality(palindrome)
        
        if verbose:
            print(f"Starting grammar improvement (initial score: {initial_score}/100)...")
            
        best_palindrome = palindrome
        best_score = initial_score
        
        # Try multiple phases of improvements
        for phase in range(max_attempts):
            improved, improved_score = self.validator.improve_palindrome_grammar(best_palindrome)
            
            if improved_score > best_score and self.is_palindrome(improved):
                best_palindrome = improved
                best_score = improved_score
                
                if verbose:
                    print(f"  Grammar improved (attempt {phase+1}): {initial_score} → {improved_score}/100")
            else:
                # If no improvement in this phase, try a different approach
                # For example, add or remove words at different positions
                words = improved.split()
                if len(words) > 5:
                    # Try removing repetitive words (especially "a" and "I")
                    for remove_word in ["a", "I", "the"]:
                        if words.count(remove_word) > 5:
                            # Remove some instances
                            indices = [i for i, w in enumerate(words) if w == remove_word]
                            if len(indices) >= 4:
                                remove_indices = random.sample(indices, 2)
                                mirror_indices = [len(words) - 1 - idx for idx in remove_indices]
                                all_indices = sorted(remove_indices + mirror_indices, reverse=True)
                                
                                test_words = words.copy()
                                for idx in all_indices:
                                    if 0 <= idx < len(test_words):
                                        test_words.pop(idx)
                                
                                test_palindrome = " ".join(test_words)
                                if self.is_palindrome(test_palindrome):
                                    test_score = self.validator.score_grammatical_quality(test_palindrome)
                                    if test_score > best_score:
                                        best_palindrome = test_palindrome
                                        best_score = test_score
                                        if verbose:
                                            print(f"  Grammar improved (word removal): {best_score}/100")
                                        break
            
            # Try to insert common sentence structures
            if phase > 5 and best_score < 70:
                # Get grammar suggestions
                suggestions = self.validator.suggest_grammar_improvements(best_palindrome)
                if suggestions and "missing determiners" in " ".join(suggestions).lower():
                    # Try to add determiners
                    words = best_palindrome.split()
                    for i, word in enumerate(words):
                        if i > 0 and i < len(words) - 1:
                            doc = self.validator.nlp(word)
                            if len(doc) == 1 and doc[0].pos_ == 'NOUN':
                                for det in ['a', 'the', 'my']:
                                    test_words = words.copy()
                                    mirror_idx = len(words) - 1 - i
                                    
                                    # Add determiner before noun
                                    test_words.insert(i, det)
                                    
                                    # Add mirror on the other side if needed
                                    if 0 <= mirror_idx < len(test_words):
                                        test_words.insert(mirror_idx + 1, det[::-1])
                                        
                                    test_palindrome = " ".join(test_words)
                                    if self.is_palindrome(test_palindrome):
                                        test_score = self.validator.score_grammatical_quality(test_palindrome)
                                        if test_score > best_score:
                                            best_palindrome = test_palindrome
                                            best_score = test_score
                                            if verbose:
                                                print(f"  Grammar improved (determiner addition): {best_score}/100")
                                            break
            
            # Try replacing words with synonyms that improve grammar
            if phase > 3 and best_score < 75:
                words = best_palindrome.split()
                
                # Focus on replacing repetitive non-essential words
                for i, word in enumerate(words):
                    if word.lower() in ["see", "look", "go", "put", "get"]:
                        # Common weak verbs - try to replace with stronger alternatives
                        alternatives = {
                            "see": ["view", "spot", "eye", "notice"],
                            "look": ["gaze", "stare", "peek", "watch"],
                            "go": ["move", "head", "travel", "walk"],
                            "put": ["place", "set", "lay", "position"],
                            "get": ["grab", "take", "fetch", "gain"]
                        }
                        
                        if word.lower() in alternatives:
                            for replacement in alternatives[word.lower()]:
                                # Try the replacement
                                test_words = words.copy()
                                test_words[i] = replacement
                                
                                # Also replace the mirror word
                                mirror_idx = len(words) - 1 - i
                                if 0 <= mirror_idx < len(words) and mirror_idx != i:
                                    # We need to reverse the replacement for the mirror position
                                    mirror_word = replacement[::-1]
                                    test_words[mirror_idx] = mirror_word
                                
                                test_palindrome = " ".join(test_words)
                                if self.is_palindrome(test_palindrome):
                                    test_score = self.validator.score_grammatical_quality(test_palindrome)
                                    if test_score > best_score:
                                        best_palindrome = test_palindrome
                                        best_score = test_score
                                        if verbose:
                                            print(f"  Grammar improved (word replacement): {best_score}/100")
                                        break
            
            # If still no improvement, try insertion of new palindromic seed phrases
            if phase > 7 and phase == max_attempts - 1 and best_score < 60:
                seed_phrases = [
                    "a man a plan a canal",
                    "never odd or even",
                    "step on no pets",
                    "was it a car or a cat I saw"
                ]
                
                for seed in seed_phrases:
                    test_palindrome = f"{seed} {best_palindrome} {' '.join(w[::-1] for w in seed.split()[::-1])}"
                    if self.is_palindrome(test_palindrome):
                        test_score = self.validator.score_grammatical_quality(test_palindrome)
                        if test_score > best_score:
                            best_palindrome = test_palindrome
                            best_score = test_score
                            if verbose:
                                print(f"  Grammar improved (seed phrase insertion): {best_score}/100")
                            break
                
        # Return the best improved palindrome
        return best_palindrome, best_score
        
    def improve_semantics(self, palindrome, max_attempts=5, verbose=False):
        """
        Improve the semantic coherence of a palindrome if semantic tools are available.
        """
        if not self.semantic_analyzer:
            return palindrome, 0
            
        initial_score = self.semantic_analyzer.evaluate_semantic_coherence(palindrome)
        
        if verbose:
            print(f"Starting semantic improvement (initial score: {initial_score}/100)...")
        
        improved_palindrome, improved_score = self.semantic_analyzer.improve_semantic_coherence(palindrome)
        
        if improved_score > initial_score and self.is_palindrome(improved_palindrome):
            if verbose:
                print(f"  Semantic coherence improved: {initial_score} → {improved_score}/100")
            return improved_palindrome, improved_score
        else:
            return palindrome, initial_score
    
    def generate_optimal_palindrome(self, target_length=250, attempts=5, improve_attempts=10, verbose=False):
        """
        Generate an optimal palindrome using multiple strategies and attempts.
        """
        # Generate candidates with multiple attempts
        best_candidate = self.generate_with_multiple_attempts(target_length, attempts, verbose)
        
        if best_candidate:
            # Apply aggressive grammar improvement
            improved_palindrome, improved_score = self.improve_grammar_aggressively(
                best_candidate, improve_attempts, verbose
            )
            
            # Verify it's still a palindrome
            if not self.is_palindrome(improved_palindrome):
                logging.warning("Improved palindrome is no longer a valid palindrome")
                return best_candidate
                
            return improved_palindrome
        else:
            logging.error("Failed to generate any valid palindrome candidates")
            return None
    
    def generate_parallel(self, target_length=250, num_threads=4, attempts_per_thread=3, verbose=False):
        """
        Generate palindromes in parallel using multiple threads, then select the best one.
        """
        candidates = []
        
        if verbose:
            print(f"Generating palindromes using {num_threads} parallel threads...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit generation tasks
            future_to_id = {
                executor.submit(self.generate_with_multiple_attempts, target_length, attempts_per_thread, False): i 
                for i in range(num_threads)
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_id):
                thread_id = future_to_id[future]
                try:
                    palindrome = future.result()
                    if palindrome:
                        score = self.validator.score_grammatical_quality(palindrome)
                        candidates.append((palindrome, score, len(palindrome)))
                        if verbose:
                            print(f"  Thread {thread_id+1}: Generated palindrome with score {score}/100, length {len(palindrome)}")
                except Exception as e:
                    logging.error(f"Thread {thread_id} generated an exception: {e}")
        
        # Select the best candidate based on grammar score and length match
        if candidates:
            # Sort by grammar score (primary) and length proximity to target (secondary)
            candidates.sort(key=lambda x: (x[1], -abs(x[2] - target_length)), reverse=True)
            best_candidate = candidates[0][0]
            best_score = candidates[0][1]
            
            if verbose:
                print(f"Selected best candidate with score {best_score}/100, length {len(best_candidate)}")
                
            # Apply grammar improvement to the best candidate
            improved_palindrome, improved_score = self.improve_grammar_aggressively(
                best_candidate, max_attempts=10, verbose=verbose
            )
            
            return improved_palindrome
        else:
            logging.error("No valid palindromes generated across all threads")
            return None

    def get_palindrome_words(self, min_length=3, max_length=8):
        """Get a list of palindrome words from NLTK corpus"""
        palindromes = []
        
        try:
            from nltk.corpus import words as nltk_words
            english_words = set(w.lower() for w in nltk_words.words() if min_length <= len(w) <= max_length)
            # Check each word to see if it's a palindrome
            for word in english_words:
                if word == word[::-1]:
                    palindromes.append(word)
        except Exception as e:
            logging.warning(f"Error accessing NLTK words: {e}")
            # Fallback to common palindromes
            palindromes = ["mom", "dad", "wow", "eye", "pop", "sis", "noon", "level", 
                          "radar", "civic", "kayak", "refer", "madam", "rotor", "stats"]
        
        return palindromes
    
    def get_reversible_words(self, min_length=2, max_length=5):
        """Get words that read as different words when reversed"""
        reversible_pairs = []
        
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
            ("straw", "warts"),
            ("deliver", "reviled"),
            ("animal", "lamina"),
            ("denim", "mined"),
            ("lager", "regal"),
            ("mood", "doom"),
            ("naps", "span"),
            ("pals", "slap"),
            ("part", "trap"),
            ("pins", "snip")
        ]
        reversible_pairs.extend(manual_pairs)
        
        # Try to find some pairs from NLTK
        try:
            # Get a sample of short words
            from nltk.corpus import words as nltk_words
            short_words = [w.lower() for w in nltk_words.words() if min_length <= len(w) <= max_length]
            word_dict = {w: True for w in short_words}
            
            for word in short_words:
                reversed_word = word[::-1]
                if reversed_word in word_dict and reversed_word != word:
                    reversible_pairs.append((word, reversed_word))
        except Exception as e:
            logging.warning(f"Error finding reversible word pairs: {e}")
        
        return reversible_pairs
    
    def generate_structured_palindrome(self, min_length=60, structured=True):
        """
        Generate a palindrome using a structured approach with better grammar.
        
        Args:
            min_length: Minimum length of the normalized palindrome
            structured: Whether to use sentence structures or just word-level mirroring
            
        Returns:
            A grammatically structured palindrome
        """
        # Get palindrome words and reversible pairs
        palindrome_words = self.get_palindrome_words()
        reversible_pairs = self.get_reversible_words()
        
        if structured:
            # Use predefined sentence structures with placeholders
            structures = [
                "{sub} {verb} {obj} {conj} {obj} {verb} {sub}",
                "{sub} {verb} {adj} {noun} {conj} {noun} {adj} {verb} {sub}",
                "{noun} {prep} {adj} {noun} {conj} {noun} {adj} {prep} {noun}",
                "{sub} {modal} {verb} {adv} {conj} {adv} {verb} {modal} {sub}",
                "{name} {verb} {obj} {prep} {noun} {conj} {noun} {prep} {obj} {verb} {name}"
            ]
            
            # Choose a structure
            structure = random.choice(structures)
            
            # Fill in placeholders with words that make the overall text a palindrome
            filled_structure = self._fill_palindrome_structure(structure, palindrome_words, reversible_pairs)
            
            # If the result is too short, try to expand it
            normalized = normalize(filled_structure)
            if len(normalized) < min_length:
                # Wrap in another palindromic structure
                filled_structure = f"A {filled_structure} a"
                
                # If still too short, add more palindromic phrases
                normalized = normalize(filled_structure)
                while len(normalized) < min_length:
                    extension = random.choice([
                        "No evil, I live on",
                        "Step on no pets",
                        "Now I see bees I won",
                        "Never odd or even",
                        "Was it a rat I saw"
                    ])
                    filled_structure = f"{extension}. {filled_structure}. {extension[::-1]}"
                    normalized = normalize(filled_structure)
            
            return filled_structure
        else:
            # Start with a palindrome seed
            seed = random.choice(palindrome_words) if palindrome_words else "radar"
            normalized = normalize(seed)
            
            # Build outward until we reach the minimum length
            while len(normalized) < min_length:
                # Choose a strategy
                strategy = random.choice(["letter", "word", "pair"])
                
                if strategy == "letter":
                    # Add a letter to both ends
                    letter = chr(random.randint(97, 122))  # a-z
                    normalized = letter + normalized + letter
                
                elif strategy == "word":
                    if palindrome_words:
                        word = normalize(random.choice(palindrome_words))
                        # Add the word to one end and its reverse to the other
                        normalized = word + normalized + word[::-1]
                
                elif strategy == "pair" and reversible_pairs:
                    # Use a reversible word pair
                    pair = random.choice(reversible_pairs)
                    normalized = normalize(pair[0]) + normalized + normalize(pair[1])
            
            # Convert normalized text to readable sentences
            return self._normalized_to_readable(normalized)
    
    def _fill_palindrome_structure(self, structure, palindrome_words, reversible_pairs):
        """Fill a structure template with palindromic words and pairs"""
        # Placeholder dictionaries
        placeholders = {
            "{sub}": ["I", "he", "she", "we", "you", "they"],
            "{verb}": ["saw", "did", "was", "won", "made", "had"],
            "{obj}": ["it", "them", "that", "this", "those"],
            "{conj}": ["and", "but", "or", "nor", "yet", "so"],
            "{adj}": ["red", "old", "new", "big", "mad", "sad"],
            "{noun}": ["dad", "mom", "sis", "dog", "cat", "rat"],
            "{prep}": ["in", "on", "at", "by", "to", "of"],
            "{adv}": ["now", "too", "so", "yet", "not"],
            "{modal}": ["can", "may", "will", "must"],
            "{name}": ["Bob", "Eve", "Ana", "Otto", "Anna"]
        }
        
        # Add palindrome words to appropriate categories
        for word in palindrome_words:
            length = len(word)
            if length <= 3:
                if random.random() < 0.5:
                    placeholders["{obj}"].append(word)
                else:
                    placeholders["{noun}"].append(word)
            else:
                placeholders["{noun}"].append(word)
        
        # Replace placeholders
        filled = structure
        
        # Get all placeholders in the structure
        all_placeholders = re.findall(r'\{[a-z]+\}', structure)
        
        # For each unique placeholder type
        unique_placeholders = set(all_placeholders)
        for placeholder in unique_placeholders:
            # Count occurrences
            count = all_placeholders.count(placeholder)
            if count == 2:  # If it appears twice, use palindromic words or pairs
                if placeholder in ["{verb}", "{noun}", "{adj}"] and reversible_pairs:
                    # Try to find a reversible pair of the right type
                    # Use a random pair if available
                    if reversible_pairs:
                        pair = random.choice(reversible_pairs)
                        # Replace first and last occurrence with the pair
                        first_pos = filled.find(placeholder)
                        last_pos = filled.rfind(placeholder)
                        if first_pos >= 0 and last_pos >= 0 and first_pos != last_pos:
                            filled = filled[:first_pos] + pair[0] + filled[first_pos+len(placeholder):last_pos] + pair[1] + filled[last_pos+len(placeholder):]
                        else:
                            # Fallback if the positions are not found correctly
                            word = random.choice(placeholders[placeholder]) if placeholders[placeholder] else "word"
                            filled = filled.replace(placeholder, word)
                    else:
                        # If no reversible pairs, use the same word for both occurrences
                        word = random.choice(placeholders[placeholder]) if placeholders[placeholder] else "word"
                        filled = filled.replace(placeholder, word)
                else:
                    # Use the same word for both occurrences (maintaining palindrome property)
                    if placeholder in placeholders and placeholders[placeholder]:
                        word = random.choice(placeholders[placeholder])
                        filled = filled.replace(placeholder, word)
                    else:
                        # Fallback if the placeholder category is empty
                        filled = filled.replace(placeholder, "word")
            else:
                # For single occurrences, just replace with a random word
                if placeholder in placeholders and placeholders[placeholder]:
                    for _ in range(count):
                        word = random.choice(placeholders[placeholder])
                        filled = filled.replace(placeholder, word, 1)
                else:
                    # Fallback if the placeholder category is empty
                    filled = filled.replace(placeholder, "word")
        
        return filled
    
    def _normalized_to_readable(self, normalized_text):
        """Convert a normalized palindrome text to readable sentences"""
        # Break into chunks that might be words
        chunks = []
        i = 0
        
        # Try to find English words in the normalized text
        english_word_set = set()
        try:
            from nltk.corpus import words as nltk_words
            english_word_set = set(w.lower() for w in nltk_words.words() if len(w) >= 2 and len(w) <= 8)
        except Exception as e:
            logging.warning(f"Error loading NLTK words: {e}")
            # Add some common short words as fallback
            english_word_set = set(["a", "an", "the", "in", "on", "at", "by", "to", "and", "but", 
                                    "or", "for", "with", "is", "am", "are", "was", "were", "be", "been"])
        
        while i < len(normalized_text):
            # Try to find the longest matching word starting at this position
            found_word = False
            for length in range(min(8, len(normalized_text) - i), 1, -1):
                potential_word = normalized_text[i:i+length]
                if potential_word in english_word_set:
                    chunks.append(potential_word)
                    i += length
                    found_word = True
                    break
            
            if not found_word:
                # If no word was found, just take a single character
                chunks.append(normalized_text[i])
                i += 1
        
        # Join chunks into words and sentences
        sentence = ""
        words = []
        current_word = ""
        
        for chunk in chunks:
            if len(chunk) > 1:  # Likely a word
                if current_word:
                    words.append(current_word)
                    current_word = ""
                words.append(chunk)
            else:  # Single character
                current_word += chunk
                if len(current_word) >= 3 or random.random() < 0.3:
                    words.append(current_word)
                    current_word = ""
        
        if current_word:
            words.append(current_word)
        
        # Build sentences with proper capitalization and punctuation
        sentences = []
        current_sentence = []
        
        for i, word in enumerate(words):
            current_sentence.append(word)
            
            # End sentence after 4-8 words or at the end
            if (i > 3 and random.random() < 0.25) or i == len(words) - 1:
                if current_sentence:
                    # Capitalize first word
                    current_sentence[0] = current_sentence[0].capitalize()
                    
                    # Add punctuation
                    punctuation = "." if random.random() < 0.7 else "?"
                    sentence = " ".join(current_sentence) + punctuation
                    sentences.append(sentence)
                    current_sentence = []
        
        return " ".join(sentences)
    
    def generate_with_llm(self, prompt_text, model="google/gemma-3n-e2b-it:free", temperature=0.7):
        """
        Generate a palindrome using an LLM with an enhanced prompt.
        
        Args:
            prompt_text: The prompt to send to the LLM
            model: The model identifier to use
            temperature: Creativity parameter (0.0-1.0)
            
        Returns:
            The generated palindrome text
        """
        if not client:
            logging.error("OpenAI client not initialized. Cannot use LLM generation.")
            return None
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=temperature,
                max_tokens=500,
                timeout=60
            )
            
            generated_text = response.choices[0].message.content.strip()
            
            # Extract the palindrome if it's embedded in explanation text
            lines = [line.strip() for line in generated_text.split("\n") if line.strip()]
            
            # Try to find a line that could be a palindrome
            palindrome_candidates = []
            for line in lines:
                # Skip short lines and obvious non-palindromes
                if len(line) < 20 or line.startswith(("Here", "This", "I created", "The palindrome")):
                    continue
                
                # Add as candidate if it contains at least one sentence
                if "." in line or "?" in line or "!" in line:
                    palindrome_candidates.append(line)
            
            # If we found candidates, check if any are palindromes
            for candidate in palindrome_candidates:
                if is_full_palindrome(candidate):
                    return candidate
            
            # If no valid palindrome was found, return the longest line
            # (we'll validate it later)
            if lines:
                longest_line = max(lines, key=len)
                if len(longest_line) > 20:
                    return longest_line
            
            return generated_text
            
        except Exception as e:
            if "429" in str(e):
                logging.error(f"Rate limit exceeded: {str(e)}")
                # Return None to signal that we should try another strategy
                return None
            else:
                logging.error(f"Error during LLM generation: {str(e)}")
                return None
    
    def create_enhanced_prompt(self, min_length=60):
        """
        Create an enhanced prompt for the LLM with better examples and instructions.
        """
        # Examples of valid palindromes
        examples = [
            "Eva, can I see bees in a cave? No evil, I live on. Eva, can I see bees in a cave?",
            "A man, a plan, a canal: Panama. A man, a plan, a canal: Panama.",
            "Rats live on no evil star. Murder for a jar of red rum.",
            "Never odd or even. Was it a car or a cat I saw?",
            "No lemons, no melon. Star comedy by demo crats."
        ]
        
        # Format examples with their normalized forms
        formatted_examples = []
        for example in examples:
            norm = normalize(example)
            formatted_examples.append(f"Example: '{example}' → Normalized: '{norm}'")
        
        # Create strategies for successful palindrome generation
        strategies = [
            "1. Start with a central palindromic word or phrase",
            "2. Build outward symmetrically, adding the same letters/words on both sides",
            "3. Create a full sentence in the first half and mirror it for the second half",
            "4. Use words that read as different words when reversed (e.g., 'live'/'evil')",
            "5. Use punctuation and spacing to make the text readable",
            "6. Make the palindrome at least 2-3 sentences long for more coherence"
        ]
        
        # Build the full prompt
        prompt = f"""Generate a character-level palindrome paragraph that reads the same forward and backward when all spaces, punctuation, and capitalization are removed.

To be clear: The ENTIRE text must be a SINGLE palindrome - not multiple separate palindromes. When all non-alphabetic characters are removed and the text is converted to lowercase, reading it forward and backward should yield the same result.

{formatted_examples[0]}
{formatted_examples[1]}

Effective strategies:
{strategies[0]}
{strategies[1]} 
{strategies[2]}
{strategies[3]}

Requirements:
1. The palindrome must be at least {min_length} characters long when normalized
2. It must be grammatically correct and somewhat meaningful
3. Use 2-3 sentences to create a more coherent paragraph
4. Make the palindrome a SINGLE CHARACTER-LEVEL palindrome for the ENTIRE text

IMPORTANT: After writing your palindrome, verify it by removing all spaces, punctuation, and converting to lowercase. Check if it reads the same forward and backward.
"""
        
        return prompt
    
    def repair_palindrome(self, text):
        """
        Attempt to repair a near-palindrome to make it valid.
        
        Args:
            text: The text to repair
            
        Returns:
            A repaired palindrome text
        """
        normalized = normalize(text)
        
        # If it's already a palindrome, return it
        if normalized == normalized[::-1]:
            return text
        
        # Strategy 1: Take the first half and mirror it
        half_length = len(normalized) // 2
        first_half = normalized[:half_length]
        
        # For odd-length palindromes, include the middle character
        if len(normalized) % 2 == 1:
            mirrored = first_half + normalized[half_length] + first_half[::-1]
        else:
            mirrored = first_half + first_half[::-1]
        
        # Convert the normalized palindrome back to readable text
        return self._normalized_to_readable(mirrored)
    
    def evaluate_palindrome(self, text):
        """
        Evaluate the quality of a palindrome based on multiple criteria.
        
        Args:
            text: The palindrome text to evaluate
            
        Returns:
            A score and a dict with evaluation metrics
        """
        normalized = normalize(text)
        
        # Basic validations
        is_palindrome = normalized == normalized[::-1]
        length = len(normalized)
        
        # Count sentences
        sentence_count = text.count('.') + text.count('?') + text.count('!')
        
        # Simple readability score (higher is better)
        words = text.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        readability = max(0, min(10, 10 - abs(avg_word_length - 5)))
        
        # Grammar score (very simple approximation)
        grammar_score = min(10, sentence_count * 2)
        
        # Overall score (weighted average)
        overall_score = 0
        if is_palindrome:
            # Length points (up to 40)
            length_score = min(40, length / 2)
            
            # Readability points (up to 30)
            readability_points = readability * 3
            
            # Grammar points (up to 30)
            grammar_points = grammar_score * 3;
            
            overall_score = length_score + readability_points + grammar_points
        
        return overall_score, {
            "is_palindrome": is_palindrome,
            "length": length,
            "sentence_count": sentence_count,
            "readability": readability,
            "grammar_score": grammar_score,
            "overall_score": overall_score
        }
    
    def generate_multi_sentence_palindrome(self, min_length=60, attempts=5, model="google/gemma-3n-e2b-it:free"):
        """
        Generate a multi-sentence palindromic text with multiple strategies.
        This is the main function that combines different approaches.
        
        Args:
            min_length: Minimum length of the normalized palindrome
            attempts: Number of generation attempts
            model: The LLM model to use
            
        Returns:
            Tuple of (palindrome_text, normalized_text, is_valid)
        """
        best_palindrome = None
        best_score = 0
        best_metrics = None
        
        # Try different generation strategies
        strategies = ["llm", "structured", "constructive"]
        rate_limited = False
        
        for attempt in range(attempts):
            logging.info(f"Attempt {attempt+1}/{attempts}")
            
            # Alternate between strategies
            strategy = strategies[attempt % len(strategies)]
            
            # Skip LLM strategy if we're rate limited
            if strategy == "llm" and rate_limited:
                logging.info("Skipping LLM generation due to rate limits")
                strategy = "structured"
            
            try:
                if strategy == "llm" and client:
                    logging.info("Using LLM generation strategy")
                    prompt = self.create_enhanced_prompt(min_length)
                    palindrome = self.generate_with_llm(prompt, model=model)
                    
                    # If we got None back, it means we hit a rate limit
                    if palindrome is None:
                        rate_limited = True
                        logging.info("Rate limited on LLM. Switching to structured generation")
                        strategy = "structured"
                        palindrome = self.generate_structured_palindrome(min_length, structured=True)
                
                elif strategy == "structured":
                    logging.info("Using structured generation strategy")
                    palindrome = self.generate_structured_palindrome(min_length, structured=True)
                
                else:  # constructive
                    logging.info("Using constructive generation strategy")
                    palindrome = self.generate_structured_palindrome(min_length, structured=False)
                
                if not palindrome or len(palindrome.strip()) < 10:
                    logging.warning("Generated palindrome is too short or empty")
                    continue
                
                # Evaluate the palindrome
                score, metrics = self.evaluate_palindrome(palindrome)
                
                # Is it a valid palindrome?
                if metrics["is_palindrome"]:
                    logging.info(f"Valid palindrome (score: {score:.1f}, length: {metrics['length']})")
                    
                    # Is it better than what we have so far?
                    if score > best_score and metrics["length"] >= min_length:
                        best_palindrome = palindrome
                        best_score = score
                        best_metrics = metrics
                        
                        # If we found a really good one, we can stop early
                        if score > 80:
                            logging.info("Found high-quality palindrome, stopping early")
                            break
                else:
                    logging.info("Not a valid palindrome, attempting repair")
                    
                    # Try to repair it
                    repaired = self.repair_palindrome(palindrome)
                    repair_score, repair_metrics = self.evaluate_palindrome(repaired)
                    
                    if repair_metrics["is_palindrome"] and repair_metrics["length"] >= min_length:
                        logging.info(f"Successful repair (score: {repair_score:.1f}, length: {repair_metrics['length']})")
                        
                        # Only update if it's better than what we have
                        if repair_score > best_score:
                            best_palindrome = repaired
                            best_score = repair_score
                            best_metrics = repair_metrics
            
            except Exception as e:
                logging.error(f"Error during generation: {str(e)}")
                continue
        
        # If we found a valid palindrome, return it
        if best_palindrome and best_metrics and best_metrics["is_palindrome"]:
            normalized = normalize(best_palindrome)
            return best_palindrome, normalized, True
        
        # If not, try one last constructive approach
        logging.info("No valid palindrome found, using fallback method")
        fallback = self.generate_structured_palindrome(min_length, structured=True)
        
        if is_full_palindrome(fallback):
            normalized = normalize(fallback)
            return fallback, normalized, True
        
        # Last resort - basic palindrome
        basic = "Eva, can I see bees in a cave? No evil, I live on. Eva, can I see bees in a cave?"
        return basic, normalize(basic), True

def main():
    """Main function to run the improved palindrome generator"""
    parser = argparse.ArgumentParser(description="Generate multi-sentence palindromic text")
    parser.add_argument("--min-length", type=int, default=60, 
                      help="Minimum length of the normalized palindrome")
    parser.add_argument("--attempts", type=int, default=5, 
                      help="Number of generation attempts")
    parser.add_argument("--model", type=str, default="google/gemma-3n-e2b-it:free", 
                      help="LLM model to use")
    parser.add_argument("--strategy", type=str, choices=["auto", "llm", "structured", "constructive"],
                      default="auto", help="Generation strategy to use")
    args = parser.parse_args()
    
    # Initialize the generator
    generator = ImprovedPalindromeGenerator()
    
    print(f"Generating multi-sentence palindrome (min length: {args.min_length}, attempts: {args.attempts})")
    
    # Generate the palindrome
    palindrome, normalized, is_valid = generator.generate_multi_sentence_palindrome(
        min_length=args.min_length,
        attempts=args.attempts,
        model=args.model
    )
    
    # Show results
    print("\n" + "="*50)
    if is_valid:
        print("✅ Valid full palindrome generated!")
    else:
        print("❌ Failed to generate a valid palindrome")
    
    print(f"\nGenerated text ({len(normalized)} chars when normalized):")
    print(f'"{palindrome}"')
    
    print("\nNormalized:")
    print(normalized)
    
    # Visual representation for shorter palindromes
    if len(normalized) <= 100:
        print("\nPalindrome visualization:")
        middle = len(normalized) // 2
        visual = normalized[:middle] + "|" + normalized[middle:]
        print(visual)
        print(" " * middle + "^" + " " * middle)
        print(" " * middle + "|" + " " * middle)
        print("Center point")

if __name__ == "__main__":
    main()
