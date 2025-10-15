from .utils import reverse_word

def generate_palindrome(word, insert=''):
    """
    Construct: word + insert + reverse(word)
    """
    return word + insert + reverse_word(word)

def generate_nested_palindrome(a, b, x=''):
    """
    Construct: a + b + x + reverse(b) + reverse(a)
    """
    return a + b + x + reverse_word(b) + reverse_word(a)
