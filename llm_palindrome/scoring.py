"""Scorers that rank letter-valid branches by how much they read as English.

FreqScorer is the dependency-free baseline used in tests. GPT2Scorer (in
lm_scoring.py) plugs a real language model into the same interface.
"""
from __future__ import annotations

from collections import Counter
from typing import Sequence


class FreqScorer:
    """Frequency + bigram-ish heuristic: prefers common words and penalizes
    immediate repetition. Deterministic and dependency-free."""

    def __init__(self, corpus_words: Sequence[str]):
        counts = Counter(w.lower() for w in corpus_words)
        total = sum(counts.values())
        self._logp = {w: (c / total) for w, c in counts.items()}

    def word_delta(self, left: tuple, right: tuple, placement: str, word: str) -> float:
        base = self._logp.get(word, 1e-6)
        seq = left if placement == "L" else right
        prev = seq[-2] if placement == "L" and len(seq) >= 2 else (
            seq[1] if placement == "R" and len(seq) >= 2 else None)
        repeat_penalty = -0.5 if prev == word else 0.0
        length_bonus = 0.05 * len(word)  # favor real words over fillers
        return base + length_bonus + repeat_penalty
