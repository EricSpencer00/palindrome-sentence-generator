"""Does this half read as an ORDER of words, or just as a bag of them?

Ranking halves by a language model's score has failed three times in this
project, and the failure has a shape: a mean logprob rewards common words. "Set
is on" scores well because "set", "is" and "on" are common, and "no it call
action" scores well for the same reason. The model is answering "are these
ordinary words" when the question is "is this ordinary English".

Subtracting the model's score for the SAME words in a different order removes
the part of the score that the vocabulary was earning. What is left is what the
word order bought — which is the thing a grammatical half has and a bag of
words does not. The measure is self-normalising, so a half of rare words is not
punished for it and a half of common ones is not paid twice.

It is still a proxy, and the rule stands: it may narrow what a person reads. It
may not decide what ships.
"""
from __future__ import annotations

import random
from typing import Sequence


def shuffles(words: Sequence[str], n: int, seed: int = 0) -> list[list[str]]:
    """`n` orderings of these words that are not the given one.

    Deterministic, because a ranking that moves between runs cannot be argued
    with. Fewer than `n` come back when the words admit fewer orderings — three
    words have five other arrangements, and asking for eight would repeat them.
    """
    rng = random.Random(seed)
    given = list(words)
    seen = {tuple(given)}
    out: list[list[str]] = []
    for _ in range(n * 12):
        if len(out) >= n:
            break
        trial = given[:]
        rng.shuffle(trial)
        if tuple(trial) not in seen:
            seen.add(tuple(trial))
            out.append(trial)
    return out


def order_gain(scores: Sequence[float]) -> float:
    """The given order's score minus the mean of its shuffles.

    `scores[0]` is the given order. Returns 0.0 when there is nothing to
    compare against, which is the honest answer for a one-word half rather than
    an infinite advantage.
    """
    if len(scores) < 2:
        return 0.0
    rest = scores[1:]
    return scores[0] - sum(rest) / len(rest)


def rank_halves(halves: Sequence[Sequence[str]], score_texts,
                n: int = 4, seed: int = 0,
                render=lambda ws: " ".join(ws).capitalize() + ".") -> list[float]:
    """Order-gain for each half, in one batched pass over every variant.

    `score_texts` takes a list of strings and returns a score per string —
    `lm_scoring.GPT2Scorer.score_texts` fits directly. Batching every half and
    every shuffle into a single call is what keeps this affordable: the cost is
    (n + 1) texts per half, not one model call per half.
    """
    texts: list[str] = []
    spans: list[tuple[int, int]] = []
    for half in halves:
        start = len(texts)
        texts.append(render(half))
        texts.extend(render(s) for s in shuffles(half, n, seed=seed))
        spans.append((start, len(texts)))
    scored = score_texts(texts)
    return [order_gain(scored[a:b]) for a, b in spans]
