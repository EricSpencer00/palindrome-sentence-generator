"""Write a short palindrome down the way a person would.

The search emits a flat run of lowercase words. `reno sir parasites set i sara
prisoner` is a valid 32-letter palindrome and nobody would read it that way;
written out it is

    Reno, sir: parasites set. I, Sara, prisoner.

Every mark in that line is free. `validator.normalize` strips case, spaces and
punctuation, so the mirror never sees any of it, which is the same licence the
catalogue takes when it writes "A man, a plan, a canal: Panama".

`textify` already spends that licence on long free-running texts by cutting at
the joins a bigram model likes least, and `experiments/punctuation_search.py`
searches it properly for texts of a few hundred words. Neither suits a
thirty-letter palindrome of six words: at that size every segmentation can be
enumerated, so there is no reason to approximate, and the marks worth having
are finer than the sentence breaks those two place.

What gets chosen
----------------
Every way of cutting the words into runs is scored, each run by the strongest
Brown-tag test it passes, and the best total wins. The mark after a run says
what the run is claiming to be:

    a sentence with subject and verb        .
    an attested sentence shape without them :   when something follows it
    locally well-formed only                ,
    nothing that parses                     ,   with no capital after it

A colon rather than a comma after a shape-only run is what makes "Reno, sir:"
read as an address rather than a list. It is used once at most, on the first
such run, because a line with two colons reads as a table.

Capitals go on the first word of each sentence and on a standalone "i", via
`spelling.spell`, which also restores apostrophes the mirror cannot see.

The letters are asserted unchanged before the result is returned. Presentation
that alters the palindrome is a bug, not a style.
"""
from __future__ import annotations

from typing import Optional, Sequence

from .spelling import spell
from .validator import normalize

SENTENCE, SHAPED, PHRASE, SHORT, NONE = 4, 3, 2, 1, 0

# Per RUN, not per word. Weighting by length lets one loose seven-word run
# outscore any reading with structure in it. But short runs cannot be free
# either, or the result is "Non, Academia, Aimed, A, Canon." — every word its
# own clause. SHORT is therefore cheap enough to be connective tissue and too
# cheap to be a strategy.
WEIGHT = {SENTENCE: 4.0, SHAPED: 2.5, PHRASE: 1.2, SHORT: 0.3, NONE: -1.0}

MIN_RUN, MAX_RUN = 1, 8


def _tier(words: Sequence[str], table, shapes, trigrams) -> int:
    from .syntax import plausible, shaped, sentence_like
    if len(words) >= 3:
        if sentence_like(words, table, shapes):
            return SENTENCE
        if shaped(words, table, shapes):
            return SHAPED
    if trigrams is not None and len(words) >= 3 and plausible(words, table, trigrams):
        return PHRASE
    # One- and two-word runs are connective tissue: "Reno", "sir", "I". They
    # are never sentences, and the tag tests do not apply to them at all.
    return SHORT if len(words) <= 2 else NONE


def segment(words: Sequence[str], table, shapes, trigrams=None
            ) -> list[tuple[list[str], int]]:
    """Best cut of `words` into (run, tier), by exhaustive dynamic programme."""
    n = len(words)
    best = [float("-inf")] * (n + 1)
    back: list[tuple[int, int]] = [(-1, NONE)] * (n + 1)
    best[0] = 0.0
    cache: dict[tuple[int, int], int] = {}
    for j in range(1, n + 1):
        for i in range(max(0, j - MAX_RUN), j):
            if best[i] == float("-inf"):
                continue
            key = (i, j)
            if key not in cache:
                cache[key] = _tier(words[i:j], table, shapes, trigrams)
            tier = cache[key]
            gain = WEIGHT[tier]
            if best[i] + gain > best[j]:
                best[j] = best[i] + gain
                back[j] = (i, tier)
    runs: list[tuple[list[str], int]] = []
    j = n
    while j > 0:
        i, tier = back[j]
        runs.append((list(words[i:j]), tier))
        j = i
    runs.reverse()
    return runs


def present(words: Sequence[str], table=None, shapes=None, trigrams=None) -> str:
    """Write `words` out with case and punctuation. Letters are unchanged."""
    words = [w for w in words if w]
    if not words:
        return ""
    if table is None:
        from .syntax import brown_tables
        table, shapes, trigrams = brown_tables()

    runs = segment(words, table, shapes, trigrams)

    out: list[str] = []
    colon_used = False
    start_of_sentence = True
    # Runs accumulated since the last full stop, and the sentences already
    # emitted. Two unrelated chunks elsewhere in the text can contain the same
    # short word run, so cutting purely on tier repeats sentences even when no
    # chunk repeats — criterion 4 in docs/NORTH-STAR.md, measured failing in
    # experiments/RESULTS-north-star-v3.md. A cut that would close a sentence
    # already used is refused and the run is absorbed into the current one
    # instead, which changes only where the marks fall.
    pending: list[str] = []
    used: set[str] = set()
    for idx, (run, tier) in enumerate(runs):
        last = idx == len(runs) - 1
        # period=False because this function picks the mark. `spell` also
        # capitalises the first word unconditionally, which is right only at a
        # sentence start; mid-sentence runs get it put back.
        text = spell(run, period=False)
        if not start_of_sentence and run[0] != "i":
            text = text[0].lower() + text[1:]
        if last:
            mark = "."
        elif tier == SENTENCE:
            mark = "."
        elif tier == SHAPED and not colon_used:
            mark, colon_used = ":", True
        else:
            mark = ","
        if mark == "." and not last:
            candidate = normalize(" ".join(pending + [text]))
            if candidate in used:
                mark = ","
        out.append(text + mark)
        pending.append(text)
        if mark in ".!?":
            # The last run of a sentence carries the mark, so the sentence key
            # is built from the unmarked texts. A duplicate can still land here
            # on the final run, where there is no later cut to defer to.
            used.add(normalize(" ".join(pending)))
            pending = []
        start_of_sentence = mark in ".!?"

    result = " ".join(out)
    assert normalize(result) == normalize(" ".join(words)), (
        "presentation changed the letters")
    return result
