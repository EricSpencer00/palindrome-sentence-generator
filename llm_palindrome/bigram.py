"""Bidirectional bigram scoring — the cheap way to make the text cohere.

Word-frequency scoring picks words that are individually common and says
nothing about whether they belong next to each other, which is exactly how a
palindrome search ends up emitting fluent-looking rubble. A bigram model fixes
the local joins at effectively no cost per candidate.

It has to work both ways round. The left half grows by PREPENDING, so when a
word is placed there the thing already known is what comes AFTER it, and the
question is P(word | successor) — a backward query. The right half grows by
appending and asks the ordinary forward question. One count table serves both;
only the conditioning side changes.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Mapping, Optional


class BigramModel:
    """Log-probabilities with a unigram backoff, in both directions."""

    # Cost applied when a pair was never observed, so an unseen join is worse
    # than a seen one but a rare word is not made unusable.
    BACKOFF_PENALTY = 3.5

    def __init__(self, pair_counts: Mapping[tuple[str, str], int],
                 unigram_counts: Mapping[str, int]):
        self._fwd: dict[str, dict[str, int]] = defaultdict(dict)
        self._bwd: dict[str, dict[str, int]] = defaultdict(dict)
        self._fwd_total: dict[str, int] = defaultdict(int)
        self._bwd_total: dict[str, int] = defaultdict(int)

        for (a, b), n in pair_counts.items():
            self._fwd[a][b] = n
            self._bwd[b][a] = n
            self._fwd_total[a] += n
            self._bwd_total[b] += n

        self._uni = dict(unigram_counts)
        self._uni_total = max(1, sum(self._uni.values()))

    # -- construction ----------------------------------------------------

    @classmethod
    def from_file(cls, path: str, vocab: Optional[Iterable[str]] = None,
                  min_count: int = 1) -> "BigramModel":
        """Load Norvig's count_2w.txt: 'word1 word2<TAB>count' per line."""
        keep = set(vocab) if vocab is not None else None
        pairs: dict[tuple[str, str], int] = {}
        uni: dict[str, int] = defaultdict(int)
        with open(path, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                try:
                    words, count = line.rstrip("\n").split("\t")
                    a, b = words.split(" ")
                    n = int(count)
                except ValueError:
                    continue
                if n < min_count:
                    continue
                a, b = a.lower(), b.lower()
                if keep is not None and (a not in keep or b not in keep):
                    continue
                pairs[(a, b)] = pairs.get((a, b), 0) + n
                uni[a] += n
                uni[b] += n
        return cls(pairs, uni)

    # -- scoring ---------------------------------------------------------

    def _unigram_logp(self, word: str) -> float:
        return math.log((self._uni.get(word, 0) + 1) / (self._uni_total + 1))

    def _conditional(self, table, totals, context: Optional[str], word: str) -> float:
        if context is None:
            return self._unigram_logp(word)
        row = table.get(context)
        if row:
            n = row.get(word)
            if n:
                return math.log(n / totals[context])
        return self._unigram_logp(word) - self.BACKOFF_PENALTY

    def forward(self, prev: Optional[str], word: str) -> float:
        """log P(word | prev) — for words appended to the right half."""
        return self._conditional(self._fwd, self._fwd_total, prev, word)

    def backward(self, word: str, nxt: Optional[str]) -> float:
        """log P(word | nxt) — for words prepended to the left half."""
        return self._conditional(self._bwd, self._bwd_total, nxt, word)
