"""Character-level constrained decoding for ordinary-order prose.

This module deliberately does not reverse words or manufacture a mirrored
surface.  Callers provide intact clause/scene fragments and a live character
residual; only fragments that consume that residual are scored.  The scorer is
an intentionally small, deterministic character n-gram model so the lane is
usable when torch/transformers or a downloaded model is unavailable.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

_LETTERS = re.compile(r"[^a-z]+")


def letters(text: str) -> str:
    return _LETTERS.sub("", text.lower())


def consume_residual(fragment: str, residual: str) -> tuple[str, bool]:
    """Consume one exact obligation, returning ``(new_residual, flipped)``.

    ``flipped`` records that the fragment overran the current obligation and
    the unmatched suffix becomes the next obligation.  No punctuation or
    word reversal is involved.
    """
    frag = letters(fragment)
    debt = letters(residual)
    if debt.startswith(frag):
        return debt[len(frag):], False
    if frag.startswith(debt):
        return frag[len(debt):], True
    return debt, False


def exact_residual_candidates(candidates: Iterable[str], residual: str) -> list[str]:
    """Return candidates whose letters overlap the live residual exactly."""
    debt = letters(residual)
    out = []
    for candidate in candidates:
        frag = letters(candidate)
        if debt.startswith(frag) or frag.startswith(debt):
            out.append(candidate)
    return out


class CharacterNgram:
    """Interpolated character n-gram score with no external model dependency."""

    def __init__(self, corpus: Iterable[str], order: int = 5,
                 alpha: float = 0.25, weight: float = 1.0):
        if order < 2:
            raise ValueError("order must be at least 2")
        self.order, self.alpha, self.weight = order, alpha, weight
        self.tables = [Counter() for _ in range(order)]
        self.totals = [Counter() for _ in range(order)]
        material = ["^" * (order - 1) + letters(row) + "$" for row in corpus]
        self.vocab = 28
        for row in material:
            for i, char in enumerate(row):
                for n in range(1, order + 1):
                    if i + 1 < n:
                        continue
                    context = row[i - n + 1:i]
                    self.tables[n - 1][(context, char)] += 1
                    self.totals[n - 1][context] += 1

    def _logp(self, history: str, char: str) -> float:
        max_n = min(self.order, len(history) + 1)
        for n in range(max_n, 0, -1):
            context = history[-(n - 1):] if n > 1 else ""
            count = self.tables[n - 1].get((context, char), 0)
            total = self.totals[n - 1].get(context, 0)
            if count or n == 1:
                return math.log((count + self.alpha) /
                                (total + self.alpha * self.vocab))
        raise AssertionError("unreachable")

    def score(self, text: str, prefix: str = "") -> float:
        """Return mean log probability per letter, conditioned on prefix."""
        stream = "^" * (self.order - 1) + letters(prefix) + letters(text)
        tail_start = len("^" * (self.order - 1) + letters(prefix))
        values = [self._logp(stream[:i], stream[i])
                  for i in range(tail_start, len(stream))]
        return self.weight * (sum(values) / len(values) if values else 0.0)


@dataclass(frozen=True)
class DecodedCandidate:
    text: str
    residual: str
    score: float
    provenance: str = "character-ngram-local-corpus"


def rank_constrained(prefix: str, candidates: Sequence[str], residual: str,
                     scorer: CharacterNgram, limit: int | None = None
                     ) -> list[DecodedCandidate]:
    """Filter by exact residual first, then rank intact fragments by fluency."""
    rows = []
    for candidate in exact_residual_candidates(candidates, residual):
        new_residual, _ = consume_residual(candidate, residual)
        rows.append(DecodedCandidate(candidate, new_residual,
                                     scorer.score(candidate, prefix)))
    rows.sort(key=lambda row: (-row.score, row.text))
    return rows if limit is None else rows[:limit]
