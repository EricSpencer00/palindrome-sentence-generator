from data_loader import load_word_list
from palindrome_core import generate_palindrome, generate_nested_palindrome
from combinator import explore_combinations
from scorer import rank_words
from validators import is_valid_word

def main():
    dictionary = load_word_list()
    # Simple palindromes
    simple_pals = [generate_palindrome(w) for w in dictionary if is_valid_word(w, dictionary)]
    # Nested palindromes (try only first 1000 for speed)
    nested = explore_combinations(list(dictionary)[:1000], dictionary)
    top100 = rank_words(simple_pals + nested)
    print("Top 100 palindromic components:")
    for w in top100:
        print(w)

if __name__ == "__main__":
    main()
