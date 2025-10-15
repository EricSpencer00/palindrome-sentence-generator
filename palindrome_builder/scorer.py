from collections import Counter

def rank_words(words):
    """Rank by frequency or appearance."""
    counter = Counter(words)
    return [w for w, _ in counter.most_common(100)]
