"""The search's scorer, with its coefficients exposed as parameters.

ZipfScorer's weights — how much word frequency counts against word length,
how hard to punish reuse — were chosen by hand. They are a policy, and the
reward that policy is being tuned against can be computed exactly. This class
is the same function with the constants pulled out so something can learn them.

The feature set is deliberately the one already in use. Learning better weights
for known features is a claim that can be checked; inventing features at the
same time would leave nothing to attribute a gain to.
"""
from __future__ import annotations

from typing import Optional, Sequence

from .scoring import adjacent

PARAMETERS = ["zipf", "length", "reuse", "adjacent_repeat", "bigram"]
DEFAULT = [1.0, 0.3, -2.0, -4.0, 0.0]  # ZipfScorer's hand-chosen values


class TunableScorer:
    """word_delta as a weighted sum over the existing features."""

    def __init__(self, weights: Optional[Sequence[float]] = None, bigrams=None):
        self.weights = list(weights) if weights is not None else list(DEFAULT)
        self.bigrams = bigrams

    def features(self, left: tuple, right: tuple, placement: str, word: str,
                 growth: str) -> list[float]:
        from wordfreq import zipf_frequency

        neighbor = adjacent(left, right, placement, growth)
        if self.bigrams is None or neighbor is None:
            bigram = 0.0
        elif growth == "prepend":
            bigram = self.bigrams.backward(word, neighbor)
        else:
            bigram = self.bigrams.forward(neighbor, word)

        return [
            zipf_frequency(word, "en"),
            float(len(word)),
            float(left.count(word) + right.count(word) - 1),
            1.0 if neighbor == word else 0.0,
            bigram,
        ]

    def word_delta(self, left: tuple, right: tuple, placement: str, word: str,
                   growth: str) -> float:
        f = self.features(left, right, placement, word, growth)
        return sum(w * x for w, x in zip(self.weights, f))

    def with_weights(self, weights: Sequence[float]) -> "TunableScorer":
        return TunableScorer(weights, self.bigrams)

    def describe(self) -> str:
        return "  ".join(f"{n}={w:+.3f}" for n, w in zip(PARAMETERS, self.weights))
