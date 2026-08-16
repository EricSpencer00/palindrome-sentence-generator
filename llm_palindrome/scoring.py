"""Scorers that rank letter-valid branches by how much they read as English.

FreqScorer is the dependency-free baseline used in tests. GPT2Scorer (in
lm_scoring.py) plugs a real language model into the same interface.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Optional, Sequence

from .search import unit_letters


def first_word(unit: str) -> str:
    """The word a unit presents to the text before it."""
    return unit.split()[0] if unit else unit


def last_word(unit: str) -> str:
    """The word a unit presents to the text after it."""
    return unit.split()[-1] if unit else unit


def unit_words(units: Iterable[str]) -> list[str]:
    """Every word in a sequence of units, phrases opened up.

    Repetition is a property of WORDS. Counting units instead would let a
    search say "york" and then "new york" and be charged for neither.
    """
    return [w for unit in units for w in unit.split()]


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
                 freq_weight: float = 0.25, length_weight: float = 0.12,
                 phrase_weight: float = 1.0, long_bonus: float = 0.0,
                 short_penalty: float = 0.0, unit_bonus=None):
        self.bg = bigrams
        self.center = center
        self.wanted = set(wanted or ())
        self.freq_weight = freq_weight
        self.length_weight = length_weight
        self.phrase_weight = phrase_weight
        self.long_bonus = long_bonus
        self.short_penalty = short_penalty
        # How good each composed unit is, as measured before it entered
        # the trie. Without this the search ranks 1500 sentences by
        # bigrams and length — neither of which is what ordered them.
        self.unit_bonus = dict(unit_bonus or {})

    def word_delta(self, left: tuple, right: tuple, placement: str, word: str,
                   growth: str) -> float:
        from wordfreq import zipf_frequency

        neighbor = adjacent(left, right, placement, growth)
        if growth == "prepend":
            # The word was placed before text that already exists, so it is
            # conditioned on what follows it. A unit meets that text at its
            # LAST word, whatever else it contains.
            if neighbor is None:
                neighbor = self.center or (right[0] if placement == "L" and right
                                           else None)
            joint = self.bg.backward(last_word(word),
                                     first_word(neighbor) if neighbor else neighbor)
        else:
            if neighbor is None:
                neighbor = self.center or (left[-1] if placement == "R" and left
                                           else None)
            joint = self.bg.forward(last_word(neighbor) if neighbor else neighbor,
                                    first_word(word))

        # The joins INSIDE a phrase belong to the phrase. They are attested by
        # construction — that is the whole reason the inventory exists — so a
        # unit that carries them must be paid for them, or the search keeps
        # preferring two loose words to the pair the corpus actually saw.
        inner = word.split()
        joint += self.phrase_weight * sum(self.bg.forward(a, b)
                                          for a, b in zip(inner, inner[1:]))

        words = unit_words(left) + unit_words(right)
        uses = sum(words.count(w) for w in inner) - len(inner)
        return (joint
                + self.freq_weight * sum(zipf_frequency(w, "en") for w in inner)
                + self.length_weight * len(unit_letters(word))
                - 2.0 * uses
                + self.long_bonus * (len(inner) - 1)
                # Only lone words pay. The penalty exists to stop the search
                # using short words as letter filler, and a multi-word unit was
                # validated as English before it entered the trie — its short
                # words are what English is 18.5% made of.
                - (self.short_penalty
                   if len(inner) == 1 and len(inner[0]) <= 2 else 0.0)
                + self.unit_bonus.get(word, 0.0)
                + (4.0 if word in self.wanted else 0.0))
