"""Bidirectional word n-gram state for center-out sentence-pair decoding."""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence


class ClauseNgramScorer:
    """Score both clauses in their actual center-out growth directions."""

    def __init__(self, sentences: Iterable[Sequence[str]], base=None,
                 order: int = 4, alpha: float = 0.1, weight: float = 1.0):
        if order < 2:
            raise ValueError("order must be at least 2")
        self.order = order
        self.alpha = alpha
        self.weight = weight
        self.base = base
        material = [tuple(word.lower() for word in sentence if word)
                    for sentence in sentences]
        self.forward = self._counts(material)
        self.backward = self._counts([tuple(reversed(s)) for s in material])
        self.vocab = max(1, len({word for sentence in material for word in sentence}))

    def _counts(self, sentences):
        tables = [Counter() for _ in range(self.order)]
        totals = [Counter() for _ in range(self.order)]
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for length in range(min(self.order - 1, i) + 1):
                    context = tuple(sentence[i - length:i])
                    tables[length][(context, word)] += 1
                    totals[length][context] += 1
        return tables, totals

    def _logp(self, model, history: Sequence[str], word: str) -> float:
        tables, totals = model
        for length in range(min(self.order - 1, len(history)), -1, -1):
            context = tuple(history[-length:]) if length else ()
            count = tables[length].get((context, word), 0)
            total = totals[length].get(context, 0)
            if count or length == 0:
                return math.log((count + self.alpha) /
                                (total + self.alpha * self.vocab))
        raise AssertionError("unreachable")

    def word_delta(self, left: tuple, right: tuple, placement: str, word: str,
                   growth: str) -> float:
        inner = word.split()
        if placement == "R":
            prior = list(right[:-1])
            model = self.forward
            added = inner
        else:
            # Prepending in natural order appends in the reversed clause.
            prior = list(reversed(left[1:]))
            model = self.backward
            added = list(reversed(inner))
        history = [part for unit in prior for part in unit.split()]
        score = 0.0
        for part in added:
            score += self._logp(model, history, part)
            history.append(part)
        base = (self.base.word_delta(left, right, placement, word, growth)
                if self.base is not None else 0.0)
        return base + self.weight * score
