"""Fit the readable axis of llm_palindrome.spectrum and report how it separates.

Anchors:
  garbage  windows of live v1 output (data/spectrum-anchors), and human
           r/palindromes titles with their words shuffled
  readable human r/palindromes titles, short Brown sentences, classics

Prints 5-fold held-out AUC and the READABLE dict to paste into spectrum.py.
"""
import glob
import json
import random
import re

import numpy as np
from nltk.corpus import brown

from llm_palindrome.spectrum import CLASSICS, Spectrum, words_of


def anchors(seed=7):
    rng = random.Random(seed)
    posts = json.load(open("data/reddit-palindromes/2026-10-09/posts.json"))["rows"]
    reddit = [r["text"] for r in posts if r["text"].isascii()
              and len(t := re.sub("[^a-z]", "", r["text"].lower())) >= 6 and t == t[::-1]]
    v1 = []
    for path in glob.glob("data/spectrum-anchors/v1-live-*.jsonl"):
        for line in open(path):
            d = json.loads(line)
            w = d["left"] + ([d["center"]] if d["center"] else []) + d["right"]
            v1 += [" ".join(w[i:i + rng.randint(4, 11)]) for i in range(0, len(w) - 10, 7)]
    shuffled = []
    for t in reddit:
        w = words_of(t)
        rng.shuffle(w)
        shuffled.append(" ".join(w))
    br = [" ".join(w for w in s if w.isalpha()) for s in brown.sents() if 4 <= len(s) <= 12]
    rng.shuffle(br)
    return {"reddit": reddit, "brown": br[:300], "classics": CLASSICS,
            "v1": v1, "shuffled": shuffled}


def main():
    sp = Spectrum()
    A = anchors()
    keys = ["syntax", "worst", "filler"]
    F = {k: np.array([[sp.features(t)[f] for f in keys] for t in v]) for k, v in A.items()}
    X = np.vstack([F["reddit"], F["brown"], F["classics"], F["v1"], F["shuffled"]])
    npos = len(F["reddit"]) + len(F["brown"]) + len(F["classics"])
    y = np.r_[np.ones(npos), np.zeros(len(X) - npos)]
    mu, sd = X.mean(0), X.std(0)
    Z = (X - mu) / sd

    def train(idx):
        w, b = np.zeros(Z.shape[1]), 0.0
        wt = np.where(y[idx] == 1, len(idx) / (2 * y[idx].sum()), len(idx) / (2 * (len(idx) - y[idx].sum())))
        for _ in range(3000):
            p = 1 / (1 + np.exp(-(Z[idx] @ w + b)))
            g = (p - y[idx]) * wt
            w -= 0.1 * (Z[idx].T @ g / len(idx) + 0.05 * w)
            b -= 0.1 * g.mean()
        return w, b

    auc = lambda p, n: np.mean([(a > c) + 0.5 * (a == c) for a in p for c in n])  # noqa: E731
    order = np.random.default_rng(0).permutation(len(y))
    held = np.zeros(len(y))
    for f in range(5):
        te = order[f::5]
        w, b = train(np.setdiff1d(order, te))
        held[te] = Z[te] @ w + b
    print(f"5-fold AUC readable vs garbage: {auc(held[y == 1], held[y == 0]):.3f}")
    w, b = train(np.arange(len(y)))
    print("READABLE =", json.dumps({"keys": keys, "mu": mu.round(4).tolist(), "sd": sd.round(4).tolist(),
                                    "w": w.round(4).tolist(), "b": round(float(b), 4)}))
    print(f"{'anchor':10s} readable  ism  position  score")
    for k, v in A.items():
        r = [sp.assess(t) for t in v]
        med = lambda f: float(np.median([x[f] for x in r]))  # noqa: E731
        print(f"{k:10s} {med('readable'):8.2f} {med('ism'):4.2f} {med('position'):9.2f} {med('score'):6.2f}")


if __name__ == "__main__":
    main()
