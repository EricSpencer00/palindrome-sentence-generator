#!/usr/bin/env python3
"""
Improved Palindrome Sentence Generator

This script combines multiple generation strategies and attempts to produce
higher-quality palindromes with better grammatical structure.
"""

import argparse
import logging
import time
import concurrent.futures
import random
from palindrome_generator import PalindromeGenerator
from grammar_palindrome_generator import GrammarPalindromeGenerator
from grammar_validator import GrammarValidator

# Configure logging
logging.basicConfig(
    filename='palindrome.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ImprovedPalindromeGenerator:
    def __init__(self):
        self.grammar_generator = GrammarPalindromeGenerator()
        self.basic_generator = PalindromeGenerator()
        self.validator = GrammarValidator()
        
    def is_palindrome(self, text):
        """Check if a string is a palindrome (ignoring spaces and punctuation)."""
        return self.grammar_generator.is_palindrome(text)
    
    def generate_with_multiple_attempts(self, target_length=250, attempts=10, verbose=False):
        """
        Generate palindromes with multiple attempts and select the best one based on
        grammar score and length proximity to target.
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
            score = self.validator.score_grammatical_quality(current)
            length_diff = abs(len(current) - target_length)
            
            # Apply weighted scoring - balancing grammar quality and length match
            combined_score = score * 0.7 - (length_diff / target_length) * 30
            
            if verbose:
                print(f"  Length: {len(current)}, Grammar score: {score}/100, Combined score: {combined_score:.1f}")
            
            # Keep track of the best palindrome
            if best_palindrome is None or combined_score > best_score:
                best_palindrome = current
                best_score = combined_score
                best_length_diff = length_diff
                
            # If we've found a very good palindrome, we can stop early
            if score > 80 and length_diff < target_length * 0.1:
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

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Generate improved palindrome paragraphs.')
    
    parser.add_argument(
        '--length', 
        type=int, 
        default=250,
        help='Target length of the palindrome paragraph (default: 250 characters).'
    )
    
    parser.add_argument(
        '--attempts', 
        type=int, 
        default=10,
        help='Number of generation attempts (default: 10).'
    )
    
    parser.add_argument(
        '--improve-attempts', 
        type=int, 
        default=15,
        help='Number of grammar improvement attempts (default: 15).'
    )
    
    parser.add_argument(
        '--parallel', 
        action='store_true',
        help='Use parallel generation for better results.'
    )
    
    parser.add_argument(
        '--threads', 
        type=int, 
        default=4,
        help='Number of parallel threads for generation (default: 4).'
    )
    
    parser.add_argument(
        '--output', 
        type=str,
        help='Output file to save the generated palindrome.'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose output.'
    )
    
    return parser.parse_args()

def main():
    """Main function to run the improved palindrome generator."""
    args = parse_arguments()
    
    start_time = time.time()
    
    if args.verbose:
        print(f"Generating improved palindrome with target length {args.length}...")
    
    generator = ImprovedPalindromeGenerator()
    
    # Choose between parallel and sequential generation
    if args.parallel:
        palindrome = generator.generate_parallel(
            target_length=args.length,
            num_threads=args.threads,
            attempts_per_thread=max(1, args.attempts // args.threads),
            verbose=args.verbose
        )
    else:
        palindrome = generator.generate_optimal_palindrome(
            target_length=args.length,
            attempts=args.attempts,
            improve_attempts=args.improve_attempts,
            verbose=args.verbose
        )
    
    end_time = time.time()
    generation_time = end_time - start_time
    
    if palindrome:
        # Display the generated palindrome
        print("\nGenerated Palindrome:")
        print("-" * 80)
        print(palindrome)
        print("-" * 80)
        
        # Display statistics
        is_palindrome = generator.is_palindrome(palindrome)
        grammar_score = generator.validator.score_grammatical_quality(palindrome)
        
        print(f"\nPalindrome length: {len(palindrome)} characters")
        print(f"Is palindrome: {is_palindrome}")
        print(f"Grammar score: {grammar_score}/100")
        print(f"Generation time: {generation_time:.2f} seconds")
        
        # Show grammar suggestions
        suggestions = generator.validator.suggest_grammar_improvements(palindrome)
        if suggestions:
            print("\nGrammar suggestions:")
            for suggestion in suggestions[:3]:  # Limit to top 3 suggestions
                print(f"- {suggestion}")
        
        # Save to output file if specified
        if args.output:
            try:
                with open(args.output, 'w') as f:
                    f.write(palindrome)
                print(f"Palindrome saved to '{args.output}'")
            except Exception as e:
                print(f"Error saving to output file: {e}")
    else:
        print("Failed to generate a valid palindrome. Please try again.")
    
if __name__ == "__main__":
    main()
