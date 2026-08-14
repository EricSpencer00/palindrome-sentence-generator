"""A judge that answers in microseconds instead of milliseconds.

GPT-2 is the metric this project trusts, but it is far too slow to be a reward
signal: reinforcement learning wants a score for every rollout, and a search
produces thousands. This model is fit to predict GPT-2's score from features
that cost nothing — attested bigrams, word frequency, repetition, word shape —
so the expensive judge can be spent on anchoring rather than on every sample.

It is a linear model on purpose. The features are already the things known to
drive the score, the training set is thousands of examples rather than
millions, and a linear fit can be inspected: `explain()` prints what the judge
believes, which is worth more here than a point of correlation.

What matters is not absolute error but **ranking**: the judge is used to choose
among candidates, so it is scored by rank correlation with GPT-2 and by how
often it picks the same winner. `evaluate()` reports both.
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Optional, Sequence

FEATURES = [
    "bias",
    "bigram_mean",        # mean log p(w_i | w_i-1) under the bigram model
    "bigram_attested",    # share of adjacent pairs the corpus has ever seen
    "zipf_mean",          # mean word frequency
    "zipf_min",           # the rarest word drags the whole text down
    "rare_rate",          # share of words below zipf 3.0
    "word_len_mean",
    "short_rate",         # share of 1-2 letter words: the search's escape hatch
    "repeat_rate",        # share of adjacent pairs that are the same word
    "reuse_rate",         # share of words used more than once anywhere
    "type_token",         # vocabulary richness
]


def features(words: Sequence[str], bigrams=None) -> list[float]:
    """Feature vector for a word sequence. No model calls, no allocation of note."""
    from wordfreq import zipf_frequency

    n = max(1, len(words))
    pairs = list(zip(words, words[1:]))
    zipfs = [zipf_frequency(w, "en") for w in words] or [0.0]
    lens = [len(w) for w in words] or [0]
    counts: dict[str, int] = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1

    if bigrams is not None and pairs:
        bg = [bigrams.forward(a, b) for a, b in pairs]
        attested = statistics.mean(
            1.0 if bigrams.forward(a, b) > bigrams.forward(None, b) else 0.0
            for a, b in pairs)
    else:
        bg, attested = [0.0], 0.0

    return [
        1.0,
        statistics.mean(bg),
        attested,
        statistics.mean(zipfs),
        min(zipfs),
        sum(1 for z in zipfs if z < 3.0) / n,
        statistics.mean(lens),
        sum(1 for w in words if len(w) <= 2) / n,
        (sum(1 for a, b in pairs if a == b) / len(pairs)) if pairs else 0.0,
        sum(1 for w in words if counts[w] > 1) / n,
        len(counts) / n,
    ]


def _solve(xtx: list[list[float]], xty: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting. Small and dependency-free."""
    n = len(xty)
    m = [row[:] + [xty[i]] for i, row in enumerate(xtx)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            continue
        m[col], m[piv] = m[piv], m[col]
        for r in range(n):
            if r == col:
                continue
            f = m[r][col] / m[col][col]
            for c in range(col, n + 1):
                m[r][c] -= f * m[col][c]
    return [m[i][n] / m[i][i] if abs(m[i][i]) > 1e-12 else 0.0 for i in range(n)]


class InstantJudge:
    def __init__(self, weights: Optional[Sequence[float]] = None, bigrams=None):
        self.weights = list(weights) if weights else [0.0] * len(FEATURES)
        self.bigrams = bigrams

    def score(self, words: Sequence[str]) -> float:
        f = features(words, self.bigrams)
        return sum(w * x for w, x in zip(self.weights, f))

    def fit(self, samples: Sequence[tuple[Sequence[str], float]],
            ridge: float = 1e-3) -> "InstantJudge":
        """Least squares with a ridge term, on standardized-enough features."""
        xs = [features(w, self.bigrams) for w, _ in samples]
        ys = [y for _, y in samples]
        n = len(FEATURES)
        xtx = [[sum(x[i] * x[j] for x in xs) + (ridge if i == j else 0.0)
                for j in range(n)] for i in range(n)]
        xty = [sum(x[i] * y for x, y in zip(xs, ys)) for i in range(n)]
        self.weights = _solve(xtx, xty)
        return self

    def evaluate(self, samples: Sequence[tuple[Sequence[str], float]]) -> dict:
        """Rank agreement with the judge being imitated, which is the point.

        Absolute error is reported too, but a judge used to pick among
        candidates only has to order them the same way.
        """
        pred = [self.score(w) for w, _ in samples]
        true = [y for _, y in samples]
        k = len(samples)
        if k < 3:
            return {"n": k}

        def rank(v):
            order = sorted(range(len(v)), key=lambda i: v[i])
            r = [0.0] * len(v)
            for pos, i in enumerate(order):
                r[i] = pos
            return r

        rp, rt = rank(pred), rank(true)
        mp, mt = statistics.mean(rp), statistics.mean(rt)
        num = sum((a - mp) * (b - mt) for a, b in zip(rp, rt))
        den = math.sqrt(sum((a - mp) ** 2 for a in rp)
                        * sum((b - mt) ** 2 for b in rt))
        spearman = num / den if den else 0.0

        # How often does the judge's pick of two candidates match GPT-2's?
        agree = tot = 0
        for i in range(0, k - 1, 2):
            j = i + 1
            if true[i] == true[j]:
                continue
            tot += 1
            agree += (pred[i] > pred[j]) == (true[i] > true[j])
        return {
            "n": k,
            "spearman": round(spearman, 4),
            "pairwise_agreement": round(agree / tot, 4) if tot else None,
            "mae": round(statistics.mean(abs(a - b) for a, b in zip(pred, true)), 4),
            "target_sd": round(statistics.pstdev(true), 4),
        }

    def explain(self) -> str:
        rows = sorted(zip(FEATURES, self.weights), key=lambda r: -abs(r[1]))
        return "\n".join(f"  {name:18s} {w:+.4f}" for name, w in rows)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(
            {"features": FEATURES, "weights": self.weights}, indent=2))

    @classmethod
    def load(cls, path: Path, bigrams=None) -> "InstantJudge":
        d = json.loads(Path(path).read_text())
        if d["features"] != FEATURES:
            raise ValueError("judge was fit on a different feature set")
        return cls(d["weights"], bigrams)
