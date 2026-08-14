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


class CoherentScorer:
    """Bigram-driven scoring for the center-out search.

    Each side is scored in the direction it actually grows: a prepended word is
    conditioned on the word that will follow it, an appended word on the word
    before it. Frequency still contributes, but only as a tiebreaker — the
    bigram term is what stops the output reading as a list of common words.
    """

    def __init__(self, bigrams, center: str = "", wanted=None,
                 freq_weight: float = 0.25, length_weight: float = 0.12):
        self.bg = bigrams
        self.center = center
        self.wanted = set(wanted or ())
        self.freq_weight = freq_weight
        self.length_weight = length_weight

    def word_delta(self, left: tuple, right: tuple, placement: str, word: str) -> float:
        from wordfreq import zipf_frequency

        if placement == "L":
            # left[0] is the word just prepended; left[1] is what follows it.
            nxt = left[1] if len(left) >= 2 else (self.center or (right[0] if right else None))
            joint = self.bg.backward(word, nxt)
        else:
            prev = right[-2] if len(right) >= 2 else (self.center or (left[-1] if left else None))
            joint = self.bg.forward(prev, word)

        uses = left.count(word) + right.count(word) - 1
        return (joint
                + self.freq_weight * zipf_frequency(word, "en")
                + self.length_weight * len(word)
                - 2.0 * uses
                + (4.0 if word in self.wanted else 0.0))
