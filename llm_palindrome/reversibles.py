"""Mirror-pairs in closed form, from words whose reverse is also a word.

Every other unit source in this project either searches or proposes. The
exhaustive hunt walks a vocabulary and ranks closures; mining reads attested
phrases and filters their mirrors; authoring asks a model and checks. All three
pay for candidates that fail.

This one cannot fail. If `rev(A)` and `rev(B)` are words, then

    A B  ||  rev(B) rev(A)

is a mirror-pair by construction — "step on" reverses letter for letter into
"no pets" because "on" reverses into "no" and "step" into "pets". Nothing is
proposed, so nothing is discarded; the mirror cost is paid entirely by how rare
reversible words are. The shipping vocabulary holds 332 of them.

The scheme was found by authoring, not by design: asked for mirror-pairs, the
model produced "slap on || no pals", "snap on || no pans", "star on || no rats"
and seven more of the same shape, which is a template wearing a disguise.
"""
from __future__ import annotations

from typing import Iterable, Iterator, Sequence

Pair = tuple[list[str], list[str]]


def semordnilaps(vocab: Iterable[str], min_letters: int = 3,
                 min_zipf: float = 0.0) -> set[tuple[str, str]]:
    """Words whose reverse is also a word, as (word, reverse) both ways round.

    Palindromic words are excluded: "level" pairs with itself, which reads as a
    stutter rather than a mirror.

    `min_zipf` matters more than it looks. The shipped lexicon is a dictionary
    intersected with a frequency list, and it still contains "tra", "oda",
    "ria", "lac" and "bom" — every one of them the reverse of a common word,
    and none of them a word a reader will accept.
    """
    words = {w.lower() for w in vocab if w and w.isalpha()}
    if min_zipf > 0:
        from wordfreq import zipf_frequency
        words = {w for w in words if zipf_frequency(w, "en") >= min_zipf}

    out: set[tuple[str, str]] = set()
    for word in words:
        mirror = word[::-1]
        if len(word) < min_letters or mirror == word or mirror not in words:
            continue
        out.add((word, mirror))
    return out


def pairs_from_reversibles(vocab: Sequence[str], min_letters: int = 3,
                           min_zipf: float = 0.0,
                           limit: int | None = None) -> Iterator[Pair]:
    """Two-word halves: (A connector, rev(connector) rev(A)).

    The connector is itself a reversible word — short ones read as connectives
    ("on"/"no", "was"/"saw", "not"/"ton"), which is why the halves come out as
    phrases rather than word lists.
    """
    reversible = semordnilaps(vocab, min_letters=1, min_zipf=min_zipf)
    # Short reversibles make the best connectors: "on" joins, "reviled" does
    # not. Sorted so the output is stable across runs.
    connectors = sorted({(a, b) for a, b in reversible if len(a) <= 3})
    heads = sorted({(a, b) for a, b in reversible if len(a) >= min_letters})

    made = 0
    for head, head_rev in heads:
        for conn, conn_rev in connectors:
            if conn == head or conn_rev == head:
                continue
            left = [head, conn]
            right = [conn_rev, head_rev]
            if len(set(left)) != len(left) or len(set(right)) != len(right):
                continue
            yield left, right
            made += 1
            if limit is not None and made >= limit:
                return
