"""Sentence-scale palindrome form using intentional mirrored refrains."""
from __future__ import annotations

import random
from typing import Sequence

from .spelling import spell
from .validator import is_palindrome, normalize

THEMES = {
    "dark": frozenset({"evil", "murder", "satan", "tragic", "prison",
                       "stressed", "desserts", "shahs", "red", "rum"}),
    "journey": frozenset({"onward", "era", "canal", "panama", "pagoda",
                          "deliver", "way", "step", "trace", "drawn"}),
    "reflection": frozenset({"interpret", "memos", "note", "dissent",
                             "assess", "action", "opposition", "test",
                             "demand", "plan"}),
    "absurd": frozenset({"dog", "cat", "animals", "tuna", "gum", "potato",
                         "baby", "cigar", "papaya", "warts"}),
}


def compose_refrain(rows: Sequence[dict], target_letters: int,
                    seed: int = 0, theme: str | None = None) -> dict:
    """Arrange distinct palindromic sentences as A B ... C ... B A.

    Exact return is explicit here, not hidden as filler: each non-central
    sentence occurs exactly twice, never adjacently.  This is the algebraically
    honest long form when readable mirror-pairs do not exist.
    """
    if theme is not None and theme not in THEMES:
        raise ValueError(f"unknown theme {theme!r}")
    eligible = [row for row in rows if is_palindrome(row["text"])
                and (theme is None
                     or set(row["text"].lower().split()) & THEMES[theme])]
    if not eligible:
        raise ValueError("need at least one palindromic sentence")
    rng = random.Random(seed)
    rng.shuffle(eligible)
    centre = eligible.pop()
    left = []
    letters = len(normalize(centre["text"]))
    for row in eligible:
        add = 2 * len(normalize(row["text"]))
        if letters + add > target_letters:
            continue
        left.append(row)
        letters += add
    sequence = left + [centre] + list(reversed(left))
    sentences = [spell(row["text"].split()) for row in sequence]
    text = " ".join(sentences)
    assert is_palindrome(text)
    counts = {row["text"]: sum(item["text"] == row["text"] for item in sequence)
              for row in sequence}
    return {"text": text, "letters": len(normalize(text)),
            "sentences": sentences, "sentence_count": len(sentences),
            "distinct_sentences": len(counts),
            "max_sentence_uses": max(counts.values()), "form": "refrain",
            "theme": theme}
