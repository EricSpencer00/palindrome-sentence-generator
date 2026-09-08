"""Cheap language-aware scoring without paying common words twice."""
from __future__ import annotations

import math
from typing import Sequence

from .scoring import adjacent, first_word, last_word
from .search import unit_letters


class RankOrderScorer:
    """Frequency-rank baseline plus bidirectional local word-order gain.

    The bigram component is a likelihood ratio: conditional log probability
    minus unigram log probability. It rewards a plausible local ordering
    instead of merely rewarding another frequent word.
    """

    def __init__(self, words: Sequence[str], bigrams=None,
                 order_weight: float = 0.0, length_weight: float = 0.20,
                 reuse_weight: float = -2.0):
        self.rank = {word: index for index, word in enumerate(words)}
        self.bigrams = bigrams
        self.order_weight = order_weight
        self.length_weight = length_weight
        self.reuse_weight = reuse_weight

    def word_delta(self, left: tuple, right: tuple, placement: str, word: str,
                   growth: str) -> float:
        inner = word.split()
        rank_term = sum(8.0 - math.log1p(self.rank.get(w, len(self.rank)))
                        for w in inner)
        used = sum((left + right).count(w) for w in inner) - len(inner)
        order = 0.0
        neighbor = adjacent(left, right, placement, growth)
        if self.bigrams is not None and neighbor is not None:
            if growth == "prepend":
                order += self.bigrams.backward_order_gain(
                    last_word(word), first_word(neighbor))
            else:
                order += self.bigrams.forward_order_gain(
                    last_word(neighbor), first_word(word))
        if self.bigrams is not None:
            order += sum(self.bigrams.forward_order_gain(a, b)
                         for a, b in zip(inner, inner[1:]))
        return (rank_term + self.length_weight * len(unit_letters(word))
                + self.reuse_weight * used + self.order_weight * order)
