import re

def normalize(text: str) -> str:
    return re.sub(r'[^a-zA-Z]', '', text).lower()

def is_full_palindrome(text: str) -> bool:
    norm = normalize(text)
    return norm == norm[::-1]
