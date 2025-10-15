from .data_loader import load_word_list
from .palindrome_core import generate_palindrome, generate_nested_palindrome
from .combinator import explore_combinations, explore_by_target
from .scorer import rank_words
from .validators import is_valid_word

def main():
    dictionary = load_word_list()
    # Simple palindromes
    simple_pals = [generate_palindrome(w) for w in dictionary if is_valid_word(w, dictionary)]
    # Efficiently scan the dictionary for palindromic decomposition patterns
    decomposed = explore_by_target(dictionary, max_a=6, max_b=6)

    # Format findings for scoring
    findings = []
    for target, ptype, parts in decomposed:
        if ptype == 'a_x_a':
            findings.append(target)
        else:
            findings.append(target)

    top100 = rank_words(simple_pals + findings)
    print("Top 100 palindromic components:")
    for w in top100:
        print(w)

if __name__ == "__main__":
    main()
