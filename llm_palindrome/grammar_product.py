"""Small whole-sentence grammar product audit.

The product accepts one fixed, role-labelled grammar path and checks its
complete normalized character tape before admitting it.  It intentionally does
not repair a mismatch or reverse a finished sentence; callers that need a
larger search space must provide independent grammar paths upstream.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ProductResult:
    candidates: tuple[str, ...]
    transitions: int
    pruned: int


def _letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def search(path: Sequence[tuple[str, str]] | Iterable[tuple[str, str]]) -> ProductResult:
    """Audit one complete role-labelled grammar path.

    A path is admitted only when its full letter tape is an exact palindrome.
    ``transitions`` counts the outside-in comparisons and ``pruned`` records
    whether a mismatch eliminated the path.  The function never constructs a
    candidate by reversing or editing the supplied words.
    """
    slots = tuple(path)
    words = tuple(word for _role, word in slots)
    rendered = " ".join(words)
    tape = _letters(rendered)
    transitions = len(tape) // 2
    mismatch = next(
        ((i, tape[i], tape[-1 - i]) for i in range(transitions) if tape[i] != tape[-1 - i]),
        None,
    )
    if not tape or mismatch is not None:
        return ProductResult(candidates=(), transitions=transitions, pruned=1 if mismatch is not None else 0)
    return ProductResult(candidates=(rendered,), transitions=transitions, pruned=0)
