"""Order mirror-pairs so the assembled paragraph reads.

Nesting pairs like brackets

    L1 L2 ... Lk  CENTER  Rk ... R2 R1

is palindromic whatever order the pairs take, so ordering is free of the letter
constraint — but not free of the mirror. Placing pair B after pair A creates
two adjacencies at once: A's left half meets B's left half on the way in, and
B's right half meets A's right half on the way out. One decision, two seams,
and they cannot be traded against each other. That is the same mirror cost the
letters pay, showing up in the sequence.

The set is small — the canon yields 23 pairs — so a greedy walk over junction
scores is enough; there is no need for the beam machinery the letter search
uses. What greedy cannot do is fix a bad seam it has already committed to, so
the ordering is a preference over readable material, not a repair for
unreadable material.
"""
from __future__ import annotations

from typing import Sequence

Pair = tuple[Sequence[str], Sequence[str]]


def junction_cost(before: Pair, after: Pair, bigrams) -> float:
    """Score of both seams created by placing `after` directly inside `before`.

    Inward, the reader crosses `before`'s left half into `after`'s left half.
    Outward — the same paragraph, later — they cross `after`'s right half into
    `before`'s right half. Scoring only the inward seam would optimise half the
    paragraph and let the other half fall where it may.
    """
    left_prev, left_next = before[0][-1], after[0][0]
    right_prev, right_next = after[1][-1], before[1][0]
    return (bigrams.forward(left_prev, left_next)
            + bigrams.forward(right_prev, right_next))


def repetition_rate(text: str) -> float:
    """Share of words that are repeats of an earlier word.

    This is what caught GPT-2 gaming the selection: optimising the LM score
    moved it from 0.356 to 0.471 while the paragraph gained nothing but "a"
    twelve times. A selection objective that can raise this is buying
    function-word density, not a subject.
    """
    words = [w for w in "".join(
        c.lower() if (c.isalpha() or c.isspace()) else " " for c in text
    ).split()]
    if not words:
        return 0.0
    return (len(words) - len(set(words))) / len(words)


def cadence_concentration(text: str) -> float:
    """Share of sentences ending in the single commonest last word.

    Bounding `repetition_rate` stopped the search buying "a" twelve times; it
    immediately started buying shape instead — "Partner is. Sign is. Warning
    is. Flower is." repeats no word past the bound and says nothing. Measured,
    8 of 28 units ended in "is" (0.29) against 2 of 14 unoptimised (0.14).

    The commonest ending rather than the top three: with three sentences the
    top three are all of them, so that version read 1.0 for any short text.
    """
    from collections import Counter

    sentences = [s.strip().split() for s in text.split(".") if s.strip()]
    endings = Counter(s[-1].lower() for s in sentences if s)
    if not endings:
        return 0.0
    return endings.most_common(1)[0][1] / sum(endings.values())


def guarded(score, max_rate: float = 0.40, max_cadence: float = 1.0):
    """Wrap a scorer so it cannot buy improvement with surface regularity.

    Hard bounds rather than penalty terms: a penalty needs a weight, and any
    weight is a knob that can be tuned until the result looks good. The bounds
    come from the unoptimised baseline's own numbers, so the search may reorder
    and reselect freely but cannot exceed the regularity it started with.

    Two doors are closed here because the search went through the second one as
    soon as the first was shut. There is no claim that these are the last two.
    """
    def scored(text: str) -> float:
        if repetition_rate(text) > max_rate:
            return float("-inf")
        if cadence_concentration(text) > max_cadence:
            return float("-inf")
        return score(text)
    return scored


def compose(pairs: Sequence[Pair], score, want: int,
            width: int = 12) -> list[Pair]:
    """Choose which units appear and in what order, scoring the whole text.

    `order_pairs` optimises seams, which are local: it cannot tell that "war as
    a" and "raw food" belong together while "roll a" and "six of" merely join
    smoothly. A paragraph's subject is a property of the selection, so the
    selection is what gets searched.

    Beam search, growing one unit at a time, scoring the rendered paragraph at
    every step. `score` takes the rendered text and returns higher-is-better;
    it is injected because every fixed proxy this project tried failed against
    judge verdicts, and the caller should own that choice.

    Each step is `width * len(pairs)` scorings, so a language model is
    affordable here only because the usable inventory is small — 26 units after
    deduplication, not 4,656.
    """
    from llm_palindrome.paragraphs import render

    want = max(0, min(want, len(pairs)))
    if want == 0:
        return []

    beam: list[tuple[float, list[Pair]]] = [(0.0, [])]
    for _ in range(want):
        nxt: list[tuple[float, list[Pair]]] = []
        for _, chosen in beam:
            taken = {" ".join(left) for left, _ in chosen}
            for cand in pairs:
                if " ".join(cand[0]) in taken:
                    continue
                grown = chosen + [cand]
                nxt.append((score(render(grown)), grown))
        if not nxt:
            break
        nxt.sort(key=lambda sc: -sc[0])
        # Deduplicate by unit SET: the beam otherwise fills with permutations
        # of one selection, and ordering is decided by the same score anyway.
        seen: set[frozenset] = set()
        beam = []
        for value, chosen in nxt:
            key = frozenset(" ".join(left) for left, _ in chosen)
            if key in seen:
                continue
            seen.add(key)
            beam.append((value, chosen))
            if len(beam) >= width:
                break
    return beam[0][1]


def order_pairs(pairs: Sequence[Pair], bigrams) -> list[Pair]:
    """Greedy ordering, outermost first.

    Starts from the pair with the best available first junction rather than an
    arbitrary one, since a greedy walk is most sensitive to where it begins.
    """
    remaining = list(pairs)
    if len(remaining) < 2:
        return remaining

    best_start, best_score = 0, float("-inf")
    for i, a in enumerate(remaining):
        for j, b in enumerate(remaining):
            if i == j:
                continue
            s = junction_cost(a, b, bigrams)
            if s > best_score:
                best_start, best_score = i, s

    out = [remaining.pop(best_start)]
    while remaining:
        scores = [junction_cost(out[-1], cand, bigrams) for cand in remaining]
        out.append(remaining.pop(scores.index(max(scores))))
    return out
