#!/usr/bin/env python3
"""
Example script demonstrating the LLM-based palindrome generation.
This example compares the different generation methods.
"""

import logging
import time
from palindrome_generator import PalindromeParagraphGenerator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compare_methods():
    """Compare different palindrome generation methods"""
    generator = PalindromeParagraphGenerator()
    
    methods = [
        ("Traditional", lambda: generator.generate(max_attempts=3, target_length=150)),
        ("Middle-Out", lambda: generator.generate_from_middle(
            center_word="a", target_length=150, max_iterations=50)),
    ]
    
    # Add bidirectional method
    methods.append(("Bidirectional", lambda: generator.generate_with_llm(
        middle="a man a plan a canal panama", target_length=150, max_attempts=3)))
    
    # Add LLM method if available
    if hasattr(generator, "llm_generator") and generator.llm_generator:
        methods.append(("LLM Bidirectional", lambda: generator.generate_with_llm(
            middle="a man a plan a canal panama", target_length=150, max_attempts=3)))
    
    results = {}
    
    for name, method in methods:
        logger.info(f"Generating palindrome using {name} method...")
        start_time = time.time()
        try:
            palindrome = method()
            end_time = time.time()
            
            # Score the palindrome
            score, details = generator.score_palindrome(palindrome)
            
            results[name] = {
                "palindrome": palindrome,
                "length": len(palindrome),
                "word_count": len(palindrome.split()),
                "time": end_time - start_time,
                "score": score,
                "is_palindrome": generator.is_palindrome(palindrome),
                "details": details
            }
            
            logger.info(f"{name} method generated {len(palindrome)} chars in {end_time - start_time:.2f}s")
        except Exception as e:
            logger.error(f"Error with {name} method: {e}")
            results[name] = {"error": str(e)}
    
    # Display comparison
    print("\n" + "="*80)
    print("PALINDROME GENERATION METHOD COMPARISON")
    print("="*80)
    
    for name, result in results.items():
        print(f"\n--- {name} Method ---")
        if "error" in result:
            print(f"Error: {result['error']}")
            continue
            
        print(f"Palindrome: {result['palindrome'][:50]}...")
        print(f"Length: {result['length']} characters")
        print(f"Word count: {result['word_count']} words")
        print(f"Generation time: {result['time']:.2f} seconds")
        print(f"Score: {result['score']:.1f}")
        
        if "details" in result and result["details"]:
            detail_metrics = ["word_diversity", "is_grammatical", "halves_differ"]
            for metric in detail_metrics:
                if metric in result["details"]:
                    value = result["details"][metric]
                    if isinstance(value, float):
                        print(f"{metric}: {value:.2f}")
                    else:
                        print(f"{metric}: {value}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    compare_methods()
