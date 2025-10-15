import re

def reverse_word(word):
    """Return reversed string."""
    return word[::-1]

def normalize(word):
    """Lowercase and strip punctuation."""
    return re.sub(r'[^a-z]', '', word.lower())
