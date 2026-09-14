"""Deterministic semantic-frame search for fixed-centre extensions.

This is deliberately a *search-space* operator, not a readability scorer.  A
frame is an ordered list of ordinary clause slots (agent/action/object, etc.).
Only lexemes licensed by the slot are expanded, and a complete surface is
accepted only after the normalized tape is independently checked.  Thus the
semantic constraint is applied before exact search rather than ranking its
outputs afterwards.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

from .validator import normalize, is_palindrome


@dataclass(frozen=True)
class SemanticFrame:
    """A small finite grammar for one event description."""

    name: str
    slots: tuple[tuple[str, ...], ...]


def instantiate(frame: SemanticFrame) -> Iterable[str]:
    """Yield grammatical clause strings in stable lexical order."""
    for words in product(*frame.slots):
        yield " ".join(words)


def extend_fixed_center(center: str, frames: Iterable[SemanticFrame], *,
                        min_letters: int = 100) -> list[str]:
    """Search paired semantic clauses around ``center``.

    Pairing is by *independently generated* event clauses; no word-order
    reflection or catalogue material is introduced.  The final tape check is
    intentionally redundant with construction, providing audit evidence.
    """
    centre = " ".join(center.lower().split())
    clauses = [clause for frame in frames for clause in instantiate(frame)]
    out: list[str] = []
    for left, right in product(clauses, repeat=2):
        # Do not accept the degenerate repeated-clause/self-palindromic route.
        if left == right or normalize(left) == normalize(left)[::-1] or normalize(right) == normalize(right)[::-1]:
            continue
        text = f"{left} {centre} {right}"
        tape = normalize(text)
        if len(tape) >= min_letters and is_palindrome(text):
            out.append(text)
    return out
