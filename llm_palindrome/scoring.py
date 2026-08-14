"""Scorers that rank letter-valid branches by how much they read as English.

FreqScorer is the dependency-free baseline used in tests. GPT2Scorer (in
lm_scoring.py) plugs a real language model into the same interface.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional, Sequence


def adjacent(left: tuple, right: tuple, placement: str, growth: str) -> Optional[str]:
    """The word now next to the one just added, in final reading order.

    `placement` says which half grew ("L"/"R"); `growth` says how ("append" /
    "prepend"). Both are needed: the two searches put the new word at opposite
    ends of the same half. Outside-in appends on the left and prepends on the
    right; center-out does the reverse. A scorer that assumes either mapping is
    reading a word from the far end of the half in one of the two searches.
    """
    seq = left if placement == "L" else right
    if len(seq) < 2:
        return None
    return seq[1] if growth == "prepend" else seq[-2]


class FreqScorer:
    """Frequency + bigram-ish heuristic: prefers common words and penalizes
    immediate repetition. Deterministic and dependency-free."""

    def __init__(self, corpus_words: Sequence[str]):
        counts = Counter(w.lower() for w in corpus_words)
        total = sum(counts.values())
        self._logp = {w: (c / total) for w, c in counts.items()}

    def word_delta(self, left: tuple, right: tuple, placement: str, word: str,
                   growth: str) -> float:
        base = self._logp.get(word, 1e-6)
        prev = adjacent(left, right, placement, growth)
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

    def word_delta(self, left: tuple, right: tuple, placement: str, word: str,
                   growth: str) -> float:
        from wordfreq import zipf_frequency

        neighbor = adjacent(left, right, placement, growth)
        if growth == "prepend":
            # The word was placed before text that already exists, so it is
            # conditioned on what follows it.
            if neighbor is None:
                neighbor = self.center or (right[0] if placement == "L" and right
                                           else None)
            joint = self.bg.backward(word, neighbor)
        else:
            if neighbor is None:
                neighbor = self.center or (left[-1] if placement == "R" and left
                                           else None)
            joint = self.bg.forward(neighbor, word)

        uses = left.count(word) + right.count(word) - 1
        return (joint
                + self.freq_weight * zipf_frequency(word, "en")
                + self.length_weight * len(word)
                - 2.0 * uses
                + (4.0 if word in self.wanted else 0.0))
