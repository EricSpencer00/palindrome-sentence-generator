"""A dictionary test, as opposed to a frequency test.

`generate.build_vocab` asks whether a string is common and safe. That is the
right question for the left half of a mined pair, which comes from an attested
English phrase. It is the wrong question for the right half, which is whatever
the mirrored letters happen to segment into: "utc", "ips", "evo", "rpm", "iot"
and "csi" are all common, none are words, and mining without this filter
produced "has pictures || ser utc ips ah".

A lemma dictionary alone over-rejects — web2 lists "pet" but not "pets", and
dropping "pets" would lose "step on no pets" — so acceptance covers regular
inflections of a headword. That admits a few non-words whose stem happens to be
one; it is a filter on obvious junk, not a grammar.
"""
from __future__ import annotations

from typing import AbstractSet, Iterator

# (suffix, replacement) pairs covering regular English inflection. The
# duplicate suffixes are deliberate: "erased" stems to "erase" by restoring a
# dropped silent e, while "walked" stems to "walk" by plain removal.
_RULES = (
    ("s", ""), ("es", ""), ("ies", "y"),
    ("ed", ""), ("ed", "e"), ("d", ""),
    ("ing", ""), ("ing", "e"),
    ("er", ""), ("er", "e"), ("est", ""),
)


# Two-letter stems are closed-class ("at", "in", "of") or abbreviations, and
# neither inflects. Without this floor "at" + the plural rule admits "ats",
# which mining then spends on a right half.
MIN_STEM = 3


def inflections(word: str) -> Iterator[str]:
    """Candidate headwords `word` could be an inflected form of."""
    for suffix, replacement in _RULES:
        if len(word) > len(suffix) and word.endswith(suffix):
            stem = word[:len(word) - len(suffix)] + replacement
            if len(stem) >= MIN_STEM:
                yield stem


def is_real_word(word: str, lexicon: AbstractSet[str]) -> bool:
    """True when `word` is a headword or a regular inflection of one."""
    if not word:
        return False
    if word in lexicon:
        return True
    return any(stem in lexicon for stem in inflections(word))


def load_lexicon(path: str) -> frozenset[str]:
    """The shipped headword list, one word per line."""
    with open(path) as handle:
        return frozenset(line.strip().lower() for line in handle
                         if line.strip())
