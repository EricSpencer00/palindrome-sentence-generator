"""Pick sentences that share a subject.

The mirror cost forces short units, short units carry no subjects, and for a
long time that was read as the cost forbidding a through-line. It does not.
Whole self-palindromic sentences pay the same 3.3 bits per letter and do carry
subjects — 8 of the canon's judged centres are a first-person narrator
doubting what they saw — and grouping those produced the first paragraph in
this project that a blind judge said was about something.

That grouping was done by reading a frequency table. This module is the
measured version: score a set of sentences by how much content they share, and
search for the best set of a given size.

Content words only. Every centre here contains "i" or "a", so a score that
counted function words would call any random sample perfectly cohesive.
"""
from __future__ import annotations

from itertools import combinations
from typing import Sequence

# Shared by everything in this inventory, so shared by nothing in particular.
FUNCTION_WORDS = frozenset("""
a i as at on of it is in to an no not was are the and or so be do for all up my
we he but its nor am me his her they that this with from by have has had been
were will would can could there their which who what when where how than then
if too very just now new one two more most some any each other such only own
""".split())


def content_words(sentence: str) -> set[str]:
    """The words in `sentence` that could be what it is about."""
    return {w for w in sentence.lower().split()
            if w.isalpha() and w not in FUNCTION_WORDS}


def cohesion(sentences: Sequence[str]) -> float:
    """Mean shared content words over every pair in the set.

    A mean rather than a total: a total grows with set size, so any search
    maximising it returns the whole inventory and calls that a theme.
    """
    if len(sentences) < 2:
        return 0.0
    bags = [content_words(s) for s in sentences]
    pairs = list(combinations(bags, 2))
    return sum(len(a & b) for a, b in pairs) / len(pairs)


def best_cluster(sentences: Sequence[str], size: int) -> list[str]:
    """The `size` sentences that share the most content.

    Seeded by each content word in turn — the sentences containing it are a
    theme by definition — then extended greedily. Seeding from the best PAIR
    instead was tried and measured: it locks onto two near-duplicate sentences
    and dilutes from there, scoring 0.476 at size 7 where the sentences sharing
    "saw" score 1.095. A strong pair plus weak additions loses to consistent
    moderate overlap, and only a seed that is already a theme avoids it.

    Exhaustive search is out of reach — 41 choose 7 is 22 million sets — but
    seeding this way is exhaustive over one-word themes, so the result can
    never be worse than the best of those.
    """
    pool = list(sentences)
    size = max(0, min(size, len(pool)))
    if size == 0:
        return []
    if size == 1:
        return pool[:1]

    def extend(chosen: list[str]) -> list[str]:
        chosen = list(chosen)
        while len(chosen) < size:
            rest = [s for s in pool if s not in chosen]
            if not rest:
                break
            chosen.append(max(rest, key=lambda s: cohesion(chosen + [s])))
        return chosen[:size]

    seeds: list[list[str]] = []
    for word in sorted({w for s in pool for w in content_words(s)}):
        group = [s for s in pool if word in content_words(s)]
        if len(group) >= 2:
            seeds.append(group[:size] if len(group) >= size else group)
    for a, b in combinations(pool, 2):
        seeds.append([a, b])

    best, best_score = pool[:size], float("-inf")
    for seed in seeds:
        candidate = extend(seed)
        score = cohesion(candidate)
        if score > best_score:
            best, best_score = candidate, score
    return best


# Openers that make a sentence a question in this inventory. The canon has no
# question marks — spacing and punctuation are invisible to the mirror, so
# every centre arrives as bare words and the opener is the only signal.
INTERROGATIVE = ("was it", "is it", "are we", "can i", "do ", "did i",
                 "who ", "what ", "shall i", "may a")


def is_question(sentence: str) -> bool:
    """Does this read as a question rather than a statement?"""
    text = sentence.lower().strip()
    return any(text.startswith(opener) for opener in INTERROGATIVE)


def order_for_refrain(sentences: Sequence[str]) -> list[str]:
    """Questions outermost, statements nearest the centre.

    A refrain reads outward from its centre in both directions, and the centre
    is the only sentence heard once. Judged blind against the two alternatives
    — statements outermost, and the two interleaved — this ordering won: the
    firmest line belongs on the turn, and doubt belongs at the edges where the
    reader enters and leaves.
    """
    questions = [s for s in sentences if is_question(s)]
    statements = [s for s in sentences if not is_question(s)]
    return questions + statements


def trim_to_theme(sentences: Sequence[str],
                  anchor: "str | None" = None) -> list[str]:
    """Drop sentences that share no content with the rest.

    A prompt matching two centres used to be padded to the requested length
    with whatever was left. Judged blind, that is worse than returning the two:
    a five-sentence refrain that stays on subject ranked above a thirteen-
    sentence one that opened on the same subject and wandered off. The
    strangers do not add to the subject, they remove it.

    Keeping anything that shares ANY content word is too weak: "stella won no
    wallets" survives beside "now sir a war is won" on the strength of "won",
    which is not what either sentence is about. The theme is its most recurrent
    content word — the same notion `best_cluster` seeds from — so that is the
    anchor, and a sentence is on theme when it carries it.

    `anchor` names the theme explicitly, which a prompt must do: asked for
    "basil", the most recurrent word in the selection was still "saw", and
    trimming to it threw away the one sentence the visitor came for.

    Never returns nothing: a request has to be answered even when the pool has
    no theme in it at all.
    """
    from collections import Counter

    if not sentences:
        return []
    if anchor:
        kept = [s for s in sentences if anchor in content_words(s)]
        if kept:
            return kept
    counts: Counter = Counter()
    for sentence in sentences:
        counts.update(content_words(sentence))
    if not counts:
        return list(sentences[:1])
    anchor, seen = counts.most_common(1)[0]
    if seen < 2:
        return list(sentences[:1])
    return [s for s in sentences if anchor in content_words(s)]
