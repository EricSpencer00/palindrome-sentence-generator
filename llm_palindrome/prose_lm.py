"""Trigram model of literary prose, read in either direction.

Trained on Brown plus the NLTK Gutenberg selection (Austen, Melville, Carroll,
Chesterton, the King James Bible, Milton, Whitman, Shakespeare): 3.8M tokens.
Commas and full stops are tokens. A palindrome ignores punctuation, so the
search may place them anywhere, and the model decides where sentences end.

The left half of a centre-out palindrome grows by prepending, so it is scored
backwards, P(w | the two tokens after it); the right half forwards. One count
table serves both. Stupid backoff: cheap, and the ranking is what matters.
"""
from __future__ import annotations

import collections
import math
import os
import pickle

from .spectrum import ROOT

CACHE = os.path.join(ROOT, "data", ".prose_lm.pkl")
BOS = "."  # a full stop doubles as the sentence boundary


def _stream():
    from nltk.corpus import brown, gutenberg
    for corpus in (brown, gutenberg):
        for sent in corpus.sents():
            out = []
            for tok in sent:
                if tok in (",", ";", ":", "--"):
                    if out and out[-1] != ",":
                        out.append(",")
                elif tok.isalpha():
                    out.append(tok.lower())
            while out and out[-1] == ",":
                out.pop()
            if len(out) >= 3:
                yield out


class ProseLM:
    def __init__(self):
        uni, bi, tri = collections.Counter(), collections.Counter(), collections.Counter()
        for s in _stream():
            seq = [BOS] + s + [BOS]
            uni.update(seq[1:])
            bi.update(zip(seq, seq[1:]))
            tri.update(zip(seq, seq[1:], seq[2:]))
        self.uni, self.total = dict(uni), sum(uni.values())
        self.bi = {k: v for k, v in bi.items() if v >= 2}
        self.tri = {k: v for k, v in tri.items() if v >= 2}
        # Context totals for each direction.
        f1, b1, f2, b2 = (collections.Counter() for _ in range(4))
        for (a, b), n in bi.items():
            f1[a] += n
            b1[b] += n
        for (a, b, c), n in self.tri.items():
            f2[(a, b)] += n
            b2[(b, c)] += n
        self.f1, self.b1, self.f2, self.b2 = dict(f1), dict(b1), dict(f2), dict(b2)

    @classmethod
    def load(cls) -> "ProseLM":
        if os.path.exists(CACHE):
            with open(CACHE, "rb") as fh:
                return pickle.load(fh)
        lm = cls()
        with open(CACHE, "wb") as fh:
            pickle.dump(lm, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return lm

    def _uni(self, w: str) -> float:
        return math.log((self.uni.get(w, 0) + 0.5) / self.total)

    def fwd(self, a: str | None, b: str | None, w: str) -> float:
        """log P(w | a b), reading forward."""
        if a is not None and b is not None:
            n = self.tri.get((a, b, w))
            if n:
                return math.log(n / self.f2[(a, b)])
        if b is not None:
            n = self.bi.get((b, w))
            if n:
                return math.log(0.4 * n / self.f1[b])
        return math.log(0.16) + self._uni(w)

    def bwd(self, w: str, b: str | None, c: str | None) -> float:
        """log P(w | b c) where b, c follow w, reading backward."""
        if b is not None and c is not None:
            n = self.tri.get((w, b, c))
            if n:
                return math.log(n / self.b2[(b, c)])
        if b is not None:
            n = self.bi.get((w, b))
            if n:
                return math.log(0.4 * n / self.b1[b])
        return math.log(0.16) + self._uni(w)

    def score(self, toks: list[str]) -> float:
        """Mean forward log-prob per token of a full text, boundaries included."""
        seq = [BOS] + toks + ([BOS] if toks[-1] != BOS else [])
        s = sum(self.fwd(seq[i - 2] if i >= 2 else None, seq[i - 1], seq[i]) for i in range(1, len(seq)))
        return s / (len(seq) - 1)


THEME_CACHE = os.path.join(ROOT, "data", ".theme_vectors.pkl")


def theme_vectors(chunk: int = 2000, min_df: int = 3):
    """Word -> {passage: tf-idf weight} over ~2,000-word passages of Brown and
    Gutenberg. Two words are related when they share passages."""
    if os.path.exists(THEME_CACHE):
        with open(THEME_CACHE, "rb") as fh:
            return pickle.load(fh)
    from nltk.corpus import brown, gutenberg
    from .spectrum import FUNCTION
    passages = []
    for corpus in (brown, gutenberg):
        for fid in corpus.fileids():
            ws = [w.lower() for w in corpus.words(fid) if w.isalpha()]
            passages += [ws[i:i + chunk] for i in range(0, len(ws), chunk)]
    df = collections.Counter()
    tfs = []
    for p in passages:
        tf = collections.Counter(w for w in p if len(w) > 2 and w not in FUNCTION)
        tfs.append(tf)
        df.update(tf.keys())
    n = len(passages)
    vec = collections.defaultdict(dict)
    for j, tf in enumerate(tfs):
        for w, c in tf.items():
            if min_df <= df[w] <= n // 4:
                vec[w][j] = (1 + math.log(c)) * math.log(n / df[w])
    for w, v in vec.items():
        norm = math.sqrt(sum(x * x for x in v.values()))
        vec[w] = {j: x / norm for j, x in v.items()}
    out = dict(vec)
    with open(THEME_CACHE, "wb") as fh:
        pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return out


def theme_bonus(vec, theme: list[str], top: int = 3000) -> dict[str, float]:
    """Cosine of each word to the theme centroid, for the `top` nearest words."""
    cen = collections.Counter()
    for w in theme:
        for j, x in vec.get(w, {}).items():
            cen[j] += x
    norm = math.sqrt(sum(x * x for x in cen.values())) or 1.0
    sims = {w: sum(x * cen.get(j, 0.0) for j, x in v.items()) / norm for w, v in vec.items()}
    return dict(sorted(sims.items(), key=lambda kv: -kv[1])[:top])
