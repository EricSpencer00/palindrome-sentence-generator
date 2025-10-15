def is_valid_word(word, dictionary):
    """Check if word is valid English."""
    return word in dictionary

def is_palindrome(s):
    """Check palindrome property."""
    s = s.lower()
    return s == s[::-1]
