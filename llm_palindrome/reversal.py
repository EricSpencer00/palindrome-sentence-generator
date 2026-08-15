"""How well a unit survives being mirrored.

A word-order palindrome reverses the whole word sequence, so every three-word
unit "A verb B" arrives in the second half as "B verb A". The assembly is
correct by construction and says nothing about whether the mirror is *true*.
Measured over the shipped banks with GPT-2, 67 of 110 units scored worse
reversed (mean -0.078 logprob per letter), and the losses cluster on animacy:
"Widows lit lamps" mirrors into "Lamps lit widows", "Priest recited prayers"
into "Prayers recited priest". The survivors take reciprocal or competitive
verbs — outlived, outran, taught, guarded, replaced — where both readings are
sayable.

The gap is expensive to compute (a language model) and stable (bank data
changes rarely), so it is measured once by `training/measure_reversal.py` and
stored beside the units. This module is the cheap half: the reader that turns
those numbers into a selection preference at request time.
"""
from __future__ import annotations

import statistics
from typing import Iterable, Mapping


def reverse_unit(unit: str) -> str:
    """The mirror image of a unit, exactly as the assembler produces it."""
    return " ".join(reversed(unit.split()))


def drop_worst(units: Iterable[str], gaps: Mapping[str, float],
               want: int, fraction: float = 0.25) -> list[str]:
    """Remove the least reversal-stable units, but never below `want`.

    A preference, not a quota. Ranking hard would serve the same handful of
    units every request, and the least stable units are disproportionately the
    stakes-bearing ones ("mothers buried sons"), which a paragraph needs. So
    only a fraction goes, and only while there is surplus to spend.

    Units with no stored measurement rank neutral rather than last: a bank edit
    should reach visitors without waiting on the next GPT-2 run.
    """
    units = list(units)
    measured = list(gaps.values())
    neutral = statistics.median(measured) if measured else 0.0

    surplus = max(0, len(units) - max(0, want))
    n_drop = min(int(len(units) * fraction), surplus)
    if n_drop <= 0:
        return units

    ranked = sorted(units, key=lambda u: gaps.get(u.lower(), neutral))
    doomed = set(ranked[:n_drop])
    return [u for u in units if u not in doomed]
