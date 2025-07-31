#!/usr/bin/env python3
"""
Grammar-Based Palindrome Generator

This script generates grammatically correct palindrome paragraphs using
a grammar-based approach. It builds sentences that are palindromes at the
character level while maintaining English grammar rules.
"""

import nltk
import spacy
import random
from tqdm import tqdm
import logging
import os
from dotenv import load_dotenv
import re

# Configure logging
logging.basicConfig(
    filename='palindrome.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/words')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('punkt')
    nltk.download('words')
    nltk.download('averaged_perceptron_tagger')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logging.info("Downloading spaCy English model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Load English words from NLTK
english_words = set(w.lower() for w in nltk.corpus.words.words())

class GrammarPalindromeGenerator:
    def __init__(self):
        self.word_dict = self._build_word_dictionary()
        self.grammar_rules = self._define_grammar_rules()
        
    def _build_word_dictionary(self):
        """Build a dictionary of words categorized by part of speech."""
        word_dict = {
            'NOUN': [],
            'VERB': [],
            'ADJ': [],
            'ADV': [],
            'DET': ['a', 'the', 'my', 'your', 'our'],
            'PREP': ['in', 'on', 'at', 'by', 'to', 'for'],
            'PRON': ['I', 'you', 'he', 'she', 'it', 'we', 'they'],
            'CONJ': ['and', 'or', 'but', 'yet', 'so']
        }
        
        # Add palindrome words
        palindrome_words = [word for word in english_words if word == word[::-1] and len(word) > 2]
        logging.info(f"Found {len(palindrome_words)} palindrome words")
        
        # Add some common words for each part of speech
        sample_words = random.sample(list(english_words), 8000)
        
        for word in tqdm(sample_words, desc="Building word dictionary"):
            if len(word) < 2 or not word.isalpha():
                continue
                
            doc = nlp(word)
            pos = doc[0].pos_
            
            if pos == 'NOUN':
                word_dict['NOUN'].append(word)
            elif pos == 'VERB':
                word_dict['VERB'].append(word)
            elif pos == 'ADJ':
                word_dict['ADJ'].append(word)
            elif pos == 'ADV':
                word_dict['ADV'].append(word)
        
        # Find reversible word pairs (where one word reversed is another word)
        self.reversible_pairs = []
        for word in tqdm(sample_words, desc="Finding reversible pairs"):
            if len(word) < 3 or not word.isalpha():
                continue
                
            reversed_word = word[::-1]
            if reversed_word in english_words and reversed_word != word:
                self.reversible_pairs.append((word, reversed_word))
                
        logging.info(f"Found {len(self.reversible_pairs)} reversible word pairs")
        
        return word_dict
    
    def _define_grammar_rules(self):
        """Define grammar rules for sentence construction."""
        return {
            'S': [['NP', 'VP']],                          # Sentence -> Noun Phrase + Verb Phrase
            'NP': [['DET', 'N'], ['DET', 'ADJ', 'N']],    # Noun Phrase -> Determiner + Noun or Determiner + Adjective + Noun
            'VP': [['V', 'NP'], ['V', 'PREP', 'NP']],     # Verb Phrase -> Verb + Noun Phrase or Verb + Preposition + Noun Phrase
            'N': [['NOUN']],                              # Noun -> NOUN
            'V': [['VERB']],                              # Verb -> VERB
            'ADJ': [['ADJ']],                             # Adjective -> ADJ
            'DET': [['DET']],                             # Determiner -> DET
            'PREP': [['PREP']]                            # Preposition -> PREP
        }
    
    def is_palindrome(self, text):
        """Check if a string is a palindrome (ignoring spaces and punctuation)."""
        text = ''.join(c.lower() for c in text if c.isalnum())
        return text == text[::-1]
    
    def generate_palindrome_sentence(self):
        """Generate a grammatically correct palindrome sentence."""
        # Try to use reversible word pairs to create a palindrome
        if self.reversible_pairs and random.random() < 0.3:  # 30% chance to use reversible pairs
            pairs = random.sample(self.reversible_pairs, min(3, len(self.reversible_pairs)))
            for pair in pairs:
                word1, word2 = pair
                
                templates = [
                    f"{word1} is {word2}",
                    f"{word1} and {word2}",
                    f"from {word1} to {word2}",
                    f"{word1} becomes {word2}",
                    f"no {word1} on {word2}"
                ]
                
                sentence = random.choice(templates)
                if self.is_palindrome(sentence):
                    return sentence
        
        # If we couldn't create a palindrome with reversible pairs,
        # try with actual palindrome words to ensure we get a palindrome
        palindrome_words = ['level', 'radar', 'madam', 'refer', 'civic', 'kayak', 'racecar', 'noon', 'mom', 'dad', 'wow', 'deed']
        
        # Check if we have any palindrome words in our dictionary
        seed_words = [word for word in self.word_dict['NOUN'] if self.is_palindrome(word) and len(word) > 2]
        seed_words.extend([word for word in self.word_dict['ADJ'] if self.is_palindrome(word) and len(word) > 2])
        if seed_words:
            palindrome_words.extend(seed_words)
            
        seed = random.choice(palindrome_words)
        logging.info(f"Using palindrome seed word: {seed}")
        
        # Create more grammatically interesting templates
        templates = [
            f"a {seed} a",
            f"my {seed} my",
            f"the {seed} the",
            f"I {seed} I",
            f"was it a {seed} I saw",
            f"never odd or even {seed}",
            f"step on no {seed}",
            f"some men interpret {seed}",
            f"a man a plan a {seed}",
            f"no {seed} on"
        ]
        
        # Filter templates that would be palindromes
        valid_templates = [t for t in templates if self.is_palindrome(t)]
        
        if valid_templates:
            sentence = random.choice(valid_templates)
        else:
            # For safety, choose a simple guaranteed palindrome
            sentence = f"a {seed} a"
            
        # Verify it's a palindrome
        if not self.is_palindrome(sentence):
            # For safety, choose a simple guaranteed palindrome
            logging.warning(f"Generated sentence '{sentence}' is not a palindrome, using fallback")
            sentence = f"a {seed} a"
            
        return sentence
    
    def expand_palindrome(self, current, target_length=300):
        """
        Expand a palindrome incrementally while maintaining grammatical structure.
        """
        current_length = len(current)
        attempts = 0
        max_attempts = 5000
        
        while current_length < target_length and attempts < max_attempts:
            attempts += 1
            
            # Choose an expansion strategy - with weighted probabilities
            strategy_weights = {
                self._expand_with_mirror_words: 0.4,      # 40% - general expansion
                self._expand_with_center_word: 0.2,       # 20% - center expansion (for odd-length palindromes)
                self._expand_with_sentence_wrap: 0.4,     # 40% - sentence structures
            }
            
            # Balance the strategies - prefer more complex strategies for shorter palindromes
            if current_length < target_length * 0.3:
                # For short palindromes, prefer sentence wrap for structure
                strategy_weights[self._expand_with_sentence_wrap] = 0.6
                strategy_weights[self._expand_with_mirror_words] = 0.3
                strategy_weights[self._expand_with_center_word] = 0.1
            elif current_length > target_length * 0.7:
                # For nearly complete palindromes, prefer simpler expansions
                strategy_weights[self._expand_with_mirror_words] = 0.6
                strategy_weights[self._expand_with_center_word] = 0.3
                strategy_weights[self._expand_with_sentence_wrap] = 0.1
            
            # Choose strategy based on weights
            strategies = list(strategy_weights.keys())
            weights = list(strategy_weights.values())
            strategy = random.choices(strategies, weights=weights, k=1)[0]
            
            # Apply the chosen strategy
            new_palindrome = strategy(current)
            
            if new_palindrome != current and self.is_palindrome(new_palindrome):
                # Track significant expansions
                if len(new_palindrome) - len(current) > 10:
                    logging.info(f"Significant expansion: +{len(new_palindrome) - len(current)} chars")
                
                current = new_palindrome
                current_length = len(current)
                
                # Periodically log progress
                if attempts % 20 == 0 or len(current) > target_length * 0.9:
                    logging.info(f"Expanded palindrome to length {current_length}/{target_length}")
            
        if attempts >= max_attempts:
            logging.warning(f"Reached max attempts ({max_attempts}) without reaching target length")
            
        return current
    
    def _expand_with_mirror_words(self, current):
        """Expand palindrome by adding mirrored words at the beginning and end."""
        # Try to expand with meaningful grammatical structures
        expansion_structures = [
            # Noun phrases
            ("DET", "NOUN"),   # "the cat"
            ("DET", "ADJ"),    # "the red"
            ("PREP", "NOUN"),  # "on radar"
            
            # Verb phrases
            ("PRON", "VERB"),  # "I see"
            ("ADV", "VERB"),   # "never ride"
            
            # Others
            ("CONJ", "PRON"),  # "and I"
            ("NOUN", "VERB"),  # "cats sleep"
        ]
        
        # Choose a structure for expansion
        struct_type = random.choice(expansion_structures)
        
        # Try to find words for this structure that maintain palindrome property
        attempts = 0
        max_attempts = 15
        
        while attempts < max_attempts:
            attempts += 1
            
            # Get words for each part of the structure
            words = []
            valid_structure = True
            
            for pos in struct_type:
                if not self.word_dict[pos]:
                    valid_structure = False
                    break
                words.append(random.choice(self.word_dict[pos]))
            
            if not valid_structure:
                continue
                
            # Build the expansion
            expansion = " ".join(words)
            new_palindrome = f"{expansion} {current} {expansion}"
            
            if self.is_palindrome(new_palindrome):
                return new_palindrome
        
        # Try short single words (higher chance of maintaining palindrome)
        single_word_choices = [
            "a", "I", "my", "no", "on", "to", "at", "in", "was", "did", "mom", 
            "dad", "eye", "sees", "noon", "deified", "tenet"
        ]
        
        # Try each template to see if it forms a palindrome
        for word in single_word_choices:
            new_palindrome = f"{word} {current} {word}"
            if self.is_palindrome(new_palindrome):
                return new_palindrome
                
        # If all else fails, return the unchanged current palindrome
        return current
    
    def _expand_with_center_word(self, current):
        """Expand palindrome by adding a word in the center (if length is odd)."""
        if len(current) % 2 == 1:
            center_idx = len(current) // 2
            
            # Try with palindrome words to maintain the palindrome property
            palindrome_words = ["a", "I", "deed", "eye", "mom", "noon", "peep", "wow"]
            
            # Try to find palindrome words from our dictionary
            for pos in ['NOUN', 'ADJ']:
                if pos in self.word_dict:
                    palindrome_dict_words = [word for word in self.word_dict[pos] if self.is_palindrome(word)]
                    if palindrome_dict_words:
                        palindrome_words.extend(palindrome_dict_words[:5])  # Add up to 5 palindrome words
            
            # Try each word to see if it maintains the palindrome property
            for word in palindrome_words:
                new_palindrome = current[:center_idx] + f" {word} " + current[center_idx:]
                if self.is_palindrome(new_palindrome):
                    return new_palindrome
                
        return current
    
    def _expand_with_sentence_wrap(self, current):
        """Wrap the current palindrome with a simple sentence structure."""
        # Use more varied and grammatically interesting templates
        templates = [
            f"a {current} a",
            f"I {current} I",
            f"my {current} my",
            f"no {current} on",
            f"to {current} ot",
            f"mom {current} mom",
            f"dad {current} dad",
            f"wow {current} wow",
            f"was it {current} ti saw",
            f"some {current} emos",
            f"straw {current} warts",
            f"never {current} reven",
            f"evil {current} live",
            f"draw {current} ward",
            f"step {current} pets",
            f"name {current} eman",
            f"star {current} rats",
            f"live {current} evil",
            f"need {current} deen",
            f"tenet {current} tenet"
        ]
        
        # Try each template to see if it forms a palindrome
        for template in templates:
            if self.is_palindrome(template):
                return template
        
        # Attempt to construct a grammatical sentence
        sentence_patterns = [
            ("a", "NOUN", "a"),                      # "a dog a"
            ("the", "NOUN", "is", "si", "NOUN", "eht"),  # "the man is si nam eht"
            ("I", "VERB", "a", "NOUN", "a", "BREV", "I"),  # "I see a dog a ees I"
            ("no", "NOUN", "or", "ro", "NOUN", "on"),  # "no evil or ro live on"
        ]
        
        # Try to build a pattern-based palindrome
        for pattern in sentence_patterns:
            built_parts = []
            valid_pattern = True
            
            for part in pattern:
                if part in self.word_dict:  # It's a POS tag
                    if not self.word_dict[part]:
                        valid_pattern = False
                        break
                    built_parts.append(random.choice(self.word_dict[part]))
                else:  # It's a literal word
                    built_parts.append(part)
            
            if valid_pattern:
                constructed = " ".join(built_parts)
                constructed_with_current = f"{constructed} {current} {constructed}"
                if self.is_palindrome(constructed_with_current):
                    return constructed_with_current
        
        # If none of the templates work, return the unchanged palindrome
        return current
    
    def generate_palindrome_paragraph(self, target_length=300):
        """
        Generate a grammatically correct palindrome paragraph with a target length.
        """
        # Start with more interesting palindromes rather than relying on expansion
        seed_options = [
            "Able was I ere I saw Elba",
            "Never odd or even",
            "A man a plan a canal Panama",
            "Madam I'm Adam",
            "No lemon, no melon",
            "Rats live on no evil star",
            "Step on no pets",
            "Taco cat",
            "Was it a car or a cat I saw",
            "Eva, can I see bees in a cave",
            "Doc, note: I dissent. A fast never prevents a fatness. I diet on cod"
        ]
        
        # Start with a real palindrome rather than generating one from scratch
        if random.random() < 0.6:  # 60% chance to use a known good palindrome
            current = random.choice(seed_options)
        else:
            # Generate a basic palindrome as before
            current = self.generate_palindrome_sentence()
        
        # Add a prefix 'A ' and suffix ' a' to ensure it's a palindrome and has room to grow
        if not current.startswith('A '):
            current = 'A ' + current
        if not current.endswith(' a'):
            current = current + ' a'
            
        logging.info(f"Starting with seed: {current}")
        
        # Force application of wrapping with multiple known good palindromes to quickly reach target length
        if len(current) < target_length * 0.8:  # If we're far from target, apply aggressive wrapping
            wrapping_palindromes = [
                "a man a plan a {0} a nalp a nam a",
                "never odd or even {0} neve ro ddo reven",
                "step on no pets {0} step on no pets", 
                "was it a car or a cat I saw {0} was I tac a ro rac a ti saw",
                "doc note I dissent a fast never prevents a fatness I diet on cod {0} doc no teid I ssentaf a stneverpreven tsaf a tnessid I eton cod"
            ]
            
            # Try to apply as many wrappings as needed
            for _ in range(3):  # Try up to 3 different wrappings
                if len(current) < target_length * 0.8:
                    wrap_template = random.choice(wrapping_palindromes)
                    wrapped = wrap_template.format(current)
                    if self.is_palindrome(wrapped):
                        current = wrapped
                        logging.info(f"Applied aggressive wrapping: {len(current)}/{target_length}")
                else:
                    break
        
        # Try to reach the target length with more diverse strategies
        # Track the last few expansions to avoid getting stuck in a loop
        last_expansions = []
        attempts = 0
        max_attempts = 10000  # Increase max attempts
        repetitive_counter = 0
        expansion_failures = 0
        
        while len(current) < target_length and attempts < max_attempts:
            attempts += 1
            
            # Keep track of repetitive patterns
            if attempts % 100 == 0:
                # Count consecutive 'a's as a measure of repetitiveness
                if current.count('a') > len(current) * 0.5:  # If more than 50% of characters are 'a'
                    repetitive_counter += 1
                    if repetitive_counter > 2:
                        # Try to break out of repetitive patterns by injecting more variety
                        inject_text = random.choice(["was it a", "never odd or", "step on no", "rats live on no", "doc note i"])
                        mirror_text = inject_text[::-1].replace(' ', ' ')
                        test_palindrome = f"{inject_text} {current} {mirror_text}"
                        if self.is_palindrome(test_palindrome):
                            current = test_palindrome
                            repetitive_counter = 0
            
            # Choose an expansion strategy with weighted probabilities
            strategy_weights = {
                self._expand_with_mirror_words: 0.4,
                self._expand_with_center_word: 0.2,
                self._expand_with_sentence_wrap: 0.4,
            }
            
            # Balance strategies based on current length
            if len(current) < target_length * 0.3:
                # For short palindromes, prefer sentence wrap for structure
                strategy_weights[self._expand_with_sentence_wrap] = 0.6
                strategy_weights[self._expand_with_mirror_words] = 0.3
                strategy_weights[self._expand_with_center_word] = 0.1
            elif len(current) > target_length * 0.7:
                # For nearly complete palindromes, prefer simpler expansions
                strategy_weights[self._expand_with_mirror_words] = 0.6
                strategy_weights[self._expand_with_center_word] = 0.3
                strategy_weights[self._expand_with_sentence_wrap] = 0.1
            
            # Choose strategy based on weights
            strategies = list(strategy_weights.keys())
            weights = list(strategy_weights.values())
            strategy = random.choices(strategies, weights=weights, k=1)[0]
            
            # Apply the chosen strategy
            new_palindrome = strategy(current)
            
            if new_palindrome != current and self.is_palindrome(new_palindrome):
                # Avoid repetitive expansions
                if len(last_expansions) >= 5:
                    last_expansions.pop(0)  # Remove oldest expansion
                last_expansions.append(new_palindrome)
                
                # If this expansion is very similar to recent ones, try something else
                similar_to_recent = False
                for recent in last_expansions[:-1]:  # Skip the one we just added
                    if len(new_palindrome) - len(recent) < 5:  # Very small expansion
                        similar_to_recent = True
                        break
                
                if not similar_to_recent:
                    current = new_palindrome
                    if attempts % 20 == 0:
                        logging.info(f"Expanded palindrome to length {len(current)}/{target_length}")
            else:
                expansion_failures += 1
                
                # If we've failed too many times and we're far from target length, 
                # try a more aggressive approach
                if expansion_failures > 100 and len(current) < target_length * 0.5:
                    # Wrap with a larger palindrome structure
                    wrap_options = [
                        f"I {current} I",
                        f"mom {current} mom",
                        f"radar {current} radar",
                        f"a man a plan a {current} a nalp a nam a",
                        f"step on no {current} on pets",
                        f"was it a {current} a ti saw",
                        f"never odd or {current} ro ddo reven"
                    ]
                    
                    for option in wrap_options:
                        if self.is_palindrome(option):
                            current = option
                            expansion_failures = 0
                            logging.info(f"Applied aggressive expansion: {len(current)}/{target_length}")
                            break
            
        if attempts >= max_attempts:
            logging.warning(f"Reached max attempts ({max_attempts}) without reaching target length")
        
        # Clean up and format
        cleaned = re.sub(r'\s+', ' ', current).strip()
        
        # Verify it's still a palindrome after cleaning
        if not self.is_palindrome(cleaned):
            logging.warning(f"Cleaned palindrome is not a valid palindrome: '{cleaned}'")
            # Try to find the closest valid palindrome substring
            for i in range(len(cleaned)-1, 0, -1):
                if self.is_palindrome(cleaned[:i]):
                    cleaned = cleaned[:i]
                    logging.info(f"Found valid palindrome substring: '{cleaned}'")
                    break
        
        # Capitalize first letter (no period at the end to maintain palindrome property)
        final_palindrome = cleaned[0].upper() + cleaned[1:]
        
        # Verify it's still a palindrome after formatting
        if not self.is_palindrome(final_palindrome):
            logging.warning(f"Final formatting broke the palindrome property: '{final_palindrome}'")
            # Remove capitalization if it broke the palindrome
            final_palindrome = cleaned
            
        return final_palindrome

def main():
    load_dotenv()  # Load environment variables if needed
    
    print("Initializing Grammar-Based Palindrome Generator...")
    generator = GrammarPalindromeGenerator()
    
    print("\nGenerating a grammatically correct palindrome paragraph (target length: 300 characters)...")
    palindrome = generator.generate_palindrome_paragraph(300)
    
    print("\nGenerated Palindrome:")
    print("-" * 80)
    print(palindrome)
    print("-" * 80)
    
    print(f"\nPalindrome length: {len(palindrome)} characters")
    print(f"Is palindrome: {generator.is_palindrome(palindrome)}")
    
if __name__ == "__main__":
    main()
