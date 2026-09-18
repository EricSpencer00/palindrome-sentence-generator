"""Retired catalogue/reflection route.

Every entry in :mod:`data/readable_palindrome_centres.json` is a complete,
individually letter-palindromic English utterance.  Reflecting a sequence of
such entries produces a longer exact palindrome while retaining visible
sentence boundaries and ordinary punctuation.  It deliberately repeats all
but the central source utterance.  This is useful as a deterministic
presentation and study control, but it is neither novel generated material
nor evidence that the complete 103-word sequence is coherent prose.  The
new-material generator remains the route to the project's actual goal. This
module is deliberately non-operational so a repeated catalogue control cannot
be rebuilt or passed off as a long readable output.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from .validator import is_palindrome, normalize


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "readable_palindrome_centres.json"


def entries() -> list[dict[str, str]]:
    raise RuntimeError(
        "reader-first catalogue reflection is retired: it is a prohibited repeated-control shortcut"
    )
    rows = json.loads(SOURCE.read_text())
    for row in rows:
        if set(row) != {"id", "text"} or not is_palindrome(row["text"]):
            raise ValueError(f"invalid reader-first centre: {row!r}")
    return rows


def compose(ids: Sequence[str]) -> dict:
    """Reflect distinct complete utterances around one centre utterance."""
    raise RuntimeError(
        "reader-first catalogue reflection is retired: it is a prohibited repeated-control shortcut"
    )
    available = {row["id"]: row for row in entries()}
    if not ids:
        raise ValueError("at least one centre id is required")
    if len(ids) != len(set(ids)):
        raise ValueError("reader-first composition requires distinct source utterances")
    try:
        chosen = [available[item] for item in ids]
    except KeyError as exc:
        raise ValueError(f"unknown centre id: {exc.args[0]!r}") from exc
    reflected = chosen + chosen[-2::-1]
    text = " ".join(row["text"] for row in reflected)
    if not is_palindrome(text):
        raise AssertionError("reader-first reflection lost exactness")
    return {
        "source_ids": list(ids),
        "rendered_units": [row["text"] for row in reflected],
        "text": text,
        "words": len(text.split()),
        "letters": len(normalize(text)),
        "exact_palindrome": True,
    }


def hundred_word_showcase() -> dict:
    """Return the deterministic repeated-utterance control composition."""
    return compose([row["id"] for row in entries()])
