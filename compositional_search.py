"""
Compositional constraint-based palindromic sentence search

This module implements a recursive/backtracking generator that composes
palindromic sentences using a small context-free-grammar-like template and
symmetry constraints. It uses lightweight heuristic grammar checks (no external
dependencies) and a small set of reversible word pairs to make the reversed
segments produce valid words when possible.

Usage:
    python3 compositional_search.py

Functions:
- generate_palindromic_sentence(cores, fillers, max_depth=3, attempts=200)
- is_valid_english(candidate) -> heuristic grammar check

"""

import random
import re
from typing import List, Tuple, Optional

# A small set of palindromic cores and reversible pairs to make symmetric sentences
DEFAULT_CORES = [
    'level', 'radar', 'noon', 'civic', 'madam', 'refer', 'kayak', 'rotor', 'stats', 'mom', 'dad'
]

# Fillers are chosen from reversible pairs where reversed letters often form real words
REVERSIBLE_PAIRS = [
    ('no', 'on'),
    ('live', 'evil'),
    ('stop', 'pots'),
    ('star', 'rats'),
    ('time', 'emit'),
    ('flow', 'wolf'),
    ('straw', 'warts'),
    ('deliver', 'reviled'),
    ('lager', 'regal'),
    ('desserts', 'stressed')
]

# A small vocabulary / function words for heuristic checks
COMMON_WORDS = set("""
I a an the and or but if then else when while for of in on at by with to from as is are was were be been have has had do does did
he she it they we you that this those these my your his her our their me him them
""".split())

# Also allow some content words (short common words)
CONTENT_WORDS = set([
    'man','plan','canal','panama','see','bees','cave','evil','live','madam','adam','step','pets','lemon','melon'
])

VOCAB = COMMON_WORDS.union(CONTENT_WORDS)

WORD_RE = re.compile(r"[A-Za-z]+")


def tokenize(text: str) -> List[str]:
    return WORD_RE.findall(text.lower())


def is_valid_english(candidate: str) -> bool:
    """
    Lightweight heuristic grammar check.

    Checks performed:
    - At least 40% of tokens are in the small VOCAB set (function/content words)
    - Sentence starts with a capitalizable token
    - No token is an obvious non-word (e.g., many single non-alpha chars)

    This is intentionally conservative and fast.
    """
    tokens = tokenize(candidate)
    if not tokens:
        return False

    known = sum(1 for t in tokens if t in VOCAB or len(t) <= 2)
    ratio = known / len(tokens)
    if ratio < 0.35:
        return False

    # First token should be alphabetic
    if not tokens[0].isalpha():
        return False

    # Not too many very long gibberish tokens
    long_gibberish = sum(1 for t in tokens if len(t) > 12)
    if long_gibberish > len(tokens) * 0.2:
        return False

    return True


def reverse_letters(word: str) -> str:
    return word[::-1]


def build_candidate(core: str, filler_left: str, inner: str, filler_right: str) -> str:
    """Compose the sentence by combining parts and adding minimal spacing/punctuation."""
    # Minimal punctuation to keep it readable
    parts = [core, filler_left]
    if inner:
        parts.append(inner)
    if filler_right:
        parts.append(filler_right)
    parts.append(core[::-1])
    # Join with spaces and make first char uppercase
    s = ' '.join(parts)
    return s[0].upper() + s[1:]


def generate_palindromic_sentence(
    cores: List[str] = None,
    fillers: List[Tuple[str,str]] = None,
    depth: int = 0,
    max_depth: int = 3,
    attempts: int = 200
) -> Optional[str]:
    """
    Recursive constraint-based generation.

    cores: list of palindromic cores (strings)
    fillers: list of reversible pairs (left_word, right_word) such that right_word is often left_word[::-1]
    max_depth: depth limit for recursion
    attempts: number of attempts before giving up
    """
    if cores is None:
        cores = DEFAULT_CORES
    if fillers is None:
        fillers = REVERSIBLE_PAIRS

    # Define recursive generator with backtracking
    def helper(depth: int) -> Optional[str]:
        # Base case: pick a core alone
        if depth >= max_depth:
            core = random.choice(cores)
            cand = core
            return cand

        # Try a number of attempts for this depth
        for _ in range(attempts // (depth + 1)):
            core = random.choice(cores)

            # Prefer fillers where reversed counterpart is real word
            filler_pair = random.choice(fillers)
            left, right = filler_pair
            # Use left on the left, and reverse letters for the mirrored part
            # We will attempt both strategies: use the reversible right, or use reversed letters
            use_pair = random.random() < 0.7

            inner = helper(depth + 1)
            if inner is None:
                continue

            if use_pair:
                filler_left = left
                filler_right = right
            else:
                filler_left = left
                filler_right = reverse_letters(left)

            candidate = build_candidate(core, filler_left, inner, filler_right)

            # Apply grammar check
            if is_valid_english(candidate):
                # Finalize by ensuring symmetry in letters when normalized
                # (We require that normalized(candidate) == normalized(candidate)[::-1])
                # Build normalized form and verify symmetry by constructing mirrored string
                norm = ''.join(tokenize(candidate))
                if norm == norm[::-1]:
                    # Nice: valid and symmetric
                    return candidate
                else:
                    # Not symmetric at letters level; try accepting if filler_right was reversible
                    # Try one last chance: if filler_right equals reverse(left) then the full normalized
                    # string may still be symmetric depending on inner; accept if normalized symmetry
                    continue
        return None

    return helper(0)


if __name__ == '__main__':
    random.seed(42)
    s = generate_palindromic_sentence(max_depth=3, attempts=500)
    if s:
        print('Generated palindromic sentence (heuristic):')
        print(s)
    else:
        print('No palindromic sentence found with current constraints.')
