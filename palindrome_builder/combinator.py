from itertools import product
from .validators import is_valid_word
from .palindrome_core import generate_nested_palindrome

def explore_combinations(words, dictionary):
    """
    Try combinations of words that may form valid English
    when assembled as palindromes.
    """
    results = []
    for a, b, x in product(words, repeat=3):
        candidate = generate_nested_palindrome(a, b, x)
        if is_valid_word(candidate, dictionary):
            results.append(candidate)
    return results
