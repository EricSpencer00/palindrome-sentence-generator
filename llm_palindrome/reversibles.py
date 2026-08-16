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


def mirror_consistent_edges(reversible: dict[str, str],
                            attested) -> dict[str, list[str]]:
    """Which reversible word may follow which, on BOTH sides of the mirror.

    A chain of reversible words w1..wn mirrors word for word into rev(wn)..
    rev(w1), so the join (wi, wi+1) appears in the left half and the join
    (rev(wi+1), rev(wi)) appears in the right. Requiring both to be attested is
    what makes the chain read forwards and backwards, and it is severe: of the
    392 words whose reverse is also a word, 57 ordered pairs survive it.
    """
    out: dict[str, list[str]] = {}
    for a in reversible:
        followers = [b for b in reversible
                     if b != a and (a, b) in attested
                     and (reversible[b], reversible[a]) in attested]
        if followers:
            out[a] = sorted(followers)
    return out


def chains(reversible: dict[str, str], edges: dict[str, list[str]],
           min_words: int = 3, max_words: int = 6) -> Iterator[Pair]:
    """Every word-aligned mirror-pair the reversible vocabulary admits.

    Exhaustive, and it does not take long, because the answer is small. This is
    the closed form taken as far as it goes: no search, no scoring, nothing
    discarded — every chain through the mirror-consistent graph, in both
    lengths that could be a clause.

    What comes out is four pairs, up to flips: "step on was || saw no pets",
    "live on was || saw no evil", "spit on was || saw no tips", "maps on was ||
    saw no spam". None of them is a sentence. That is the whole yield of
    word-aligned construction, and it is why the material has to come from the
    walk, where the two halves may be segmented differently.
    """
    def walk(path: list[str]) -> Iterator[Pair]:
        if min_words <= len(path) <= max_words:
            left = list(path)
            right = [reversible[w] for w in reversed(path)]
            if not set(left) & set(right) and len(set(right)) == len(right):
                yield left, right
        if len(path) >= max_words:
            return
        for follower in edges.get(path[-1], ()):
            if follower not in path:
                yield from walk(path + [follower])

    for word in sorted(reversible):
        yield from walk([word])
