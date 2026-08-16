"""Scoring that can see the debt.

Every scorer in this project reads the text placed so far and nothing else.
`word_delta(left, right, placement, word, growth)` has no access to
`state.overhang`, so a word that reads beautifully and leaves a run of letters
the other half can never spell is indistinguishable, to the scorer, from one
that leaves a clean debt.

That blindness explains two measurements that otherwise look unrelated. Raising
the vocabulary from 30k to 100k takes the share of English whose mirror can be
spelled at all from 25% to 100%, and moves generated coherence not at all.
Putting GPT-2 in the search loop costs 7x and moves it not at all either
(-0.058 against -0.058). Both improve the ranking of the half you can see, and
in a palindrome the bill is always paid on the half you cannot.

The fix is one step of lookahead. Before taking a word, ask how many ways the
debt it leaves could be repaid — a question the trie can already answer, and
answers again and again for the same few thousand overhangs, so it is cached.

This is deliberately not a language model. A model would rank the repayments;
the search's problem is that most branches have no repayment to rank.
"""
from __future__ import annotations

import math
from typing import Optional

from .search import WordTries


class DebtIndex:
    """How repayable is a run of owed letters?

    `options` counts the units that could consume the overhang from either
    direction. Zero means the branch is dead however well it reads. The count
    is capped because the difference between 200 ways and 800 ways does not
    matter — the difference between none and some is the whole signal.
    """

    def __init__(self, tries: WordTries, limit: int = 64):
        self.tries = tries
        self.limit = limit
        self.cache: dict[str, int] = {}

    def options(self, overhang: str) -> int:
        hit = self.cache.get(overhang)
        if hit is not None:
            return hit
        if not overhang:
            n = self.limit           # nothing owed: every unit is available
        else:
            n = len(self.tries.left_candidates(overhang, self.limit))
            if n == 0:
                # The debt may be owed by the other side, which consumes it
                # mirrored — a dead end forwards is not a dead end at all.
                n = len(self.tries.right_candidates(overhang[::-1], self.limit))
        self.cache[overhang] = n
        return n


class OverhangAware:
    """Wraps a scorer so the debt a word leaves counts against it.

    `log1p` rather than the raw count: the search needs to tell a dead branch
    from a live one, not to chase the branch with the most options, and paying
    linearly for options is how a search ends up preferring whatever letter
    happens to be most repayable regardless of what it says.
    """

    wants_overhang = True

    def __init__(self, base, debt: DebtIndex, debt_weight: float = 1.0,
                 dead_penalty: float = 12.0):
        self.base = base
        self.debt = debt
        self.debt_weight = debt_weight
        self.dead_penalty = dead_penalty

    def prepare(self, beam):
        if hasattr(self.base, "prepare"):
            self.base.prepare(beam)

    def word_delta(self, left, right, placement: str, word: str, growth: str,
                   overhang: Optional[str] = None) -> float:
        score = self.base.word_delta(left, right, placement, word, growth)
        if overhang is None or not self.debt_weight:
            return score
        options = self.debt.options(overhang)
        if options == 0:
            return score - self.dead_penalty
        return score + self.debt_weight * math.log1p(options)
