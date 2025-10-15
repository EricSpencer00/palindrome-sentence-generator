from itertools import product
from .validators import is_valid_word
from .palindrome_core import generate_nested_palindrome, generate_palindrome


def explore_combinations(words, dictionary):
    """
    Naive Cartesian exploration (kept for reference). Very slow for large vocabularies.
    """
    results = []
    for a, b, x in product(words, repeat=3):
        candidate = generate_nested_palindrome(a, b, x)
        if is_valid_word(candidate, dictionary):
            results.append(candidate)
    return results


def explore_by_target(dictionary, max_a=6, max_b=6):
    """
    Efficiently scan target words in `dictionary` and try to decompose them into
    patterns:

      1) a + x + reverse(a)
      2) a + b + x + reverse(b) + reverse(a)

    This avoids cubic search by testing possible splits of each target word.

    Returns list of tuples: (target_word, pattern_type, parts)
    where parts is a dict containing discovered a, b, x.
    """
    results = []
    words = list(dictionary)

    for target in words:
        t = target.lower()
        L = len(t)

        # Try pattern 1: a + x + reverse(a)
        # iterate possible length of a
        for la in range(1, min(max_a, L//2) + 1):
            left = t[:la]
            right = t[-la:]
            if left[::-1] == right:
                x = t[la:-la]
                # check semantic validity: x may be empty or a valid word
                results.append((target, 'a_x_a', {'a': left, 'x': x}))
                # we can continue checking smaller la as well

        # Try pattern 2: a + b + x + reverse(b) + reverse(a)
        # pick lengths for a and b
        for la in range(1, min(max_a, L//2) + 1):
            for lb in range(1, min(max_b, (L - 2*la)//2 + 1)):
                if 2*la + 2*lb > L:
                    continue
                a_left = t[:la]
                a_right = t[-la:]
                if a_left[::-1] != a_right:
                    continue

                # b is next lb chars after a
                b_left = t[la:la+lb]
                b_right = t[-la-lb:-la]
                if b_left[::-1] != b_right:
                    continue

                x = t[la+lb: L - la - lb]
                results.append((target, 'a_b_x_b_a', {'a': a_left, 'b': b_left, 'x': x}))

    return results
