"""A small amount of human preference, kept in its place.

GPT-2 fluency is a proxy for readability, and it is wrong in specific ways: it
likes frequent words, so it will take a string of common monosyllables over a
sentence that actually says something. A person can tell those apart, and it
does not take many judgements to correct for a bias that consistent.

This is deliberately not a full preference-learning pipeline. There will be
tens of pairs, not thousands, so the model is a logistic fit on the same
features the instant judge already uses, heavily regularized, and reported by
held-out pair accuracy. If that accuracy is near chance, the honest conclusion
is that the preferences carry no signal beyond the judge and the correction
should be left off — `fit` says so rather than shipping a flattering number.

    python training/preferences.py ask  --n 30     # writes pairs to rate
    # ...edit runs/preferences.json, setting "prefer" to "a" or "b"...
    python training/preferences.py fit
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

from llm_palindrome.bigram import BigramModel
from llm_palindrome.generate import build_vocab
from llm_palindrome.instant_judge import FEATURES, InstantJudge, features
from llm_palindrome.textify import textify


def ask(args) -> None:
    """Choose pairs the judge cannot separate but a reader might.

    Random pairing wastes the human. Two palindromes from nearby seeds share a
    prefix and most of their words, so there is nothing to prefer; two from
    different arms differ so obviously that the answer carries no information
    GPT-2 does not already have. The informative pairs are the ones that are
    close in GPT-2's score and *unalike in their words* — where the model is
    indifferent and a person is not.

    Restricted to the real scorer's arm. The weakened arms exist to give the
    instant judge a range to fit; asking a person to choose between two piles
    of abbreviations teaches nothing about the text this project produces, and
    "close in score" finds exactly those pairs if it is allowed to.
    """
    rows = [r for r in json.loads(args.samples.read_text()) if r.get("words")]
    scored = [r for r in rows if r.get("gpt2") is not None
              and (args.arm == "any" or r.get("arm") == args.arm)]
    if not scored:
        raise SystemExit(f"no samples from arm {args.arm!r} in {args.samples}")
    rng = random.Random(args.seed)
    rng.shuffle(scored)

    def overlap(a, b) -> float:
        sa, sb = set(a["words"]), set(b["words"])
        return len(sa & sb) / max(1, len(sa | sb))

    candidates = []
    for i, a in enumerate(scored):
        for b in scored[i + 1:i + 40]:
            gap = abs(a["gpt2"] - b["gpt2"])
            if gap > args.max_score_gap:
                continue
            ov = overlap(a, b)
            if ov > args.max_overlap:
                continue
            candidates.append((ov, gap, a, b))

    candidates.sort(key=lambda c: (c[0], c[1]))  # least alike first
    used, pairs = set(), []
    for ov, gap, a, b in candidates:
        ka, kb = id(a), id(b)
        if ka in used or kb in used:
            continue
        used |= {ka, kb}
        pairs.append({
            "a": {"words": a["words"], "text": textify(a["words"]),
                  "gpt2": a.get("gpt2")},
            "b": {"words": b["words"], "text": textify(b["words"]),
                  "gpt2": b.get("gpt2")},
            "word_overlap": round(ov, 3),
            "gpt2_gap": round(gap, 4),
            "prefer": None,  # set to "a" or "b"; leave null to skip
        })
        if len(pairs) >= args.n:
            break
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(pairs, indent=2))
    print(f"wrote {len(pairs)} pairs to {args.out}")
    print('Set "prefer" to "a" or "b" on the ones you have an opinion about. '
          'Leave the rest null — a skipped pair is more useful than a guessed one.')


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def fit(args) -> None:
    pairs = json.loads(args.pairs.read_text())
    rated = [p for p in pairs if p.get("prefer") in ("a", "b")]
    if len(rated) < 8:
        print(f"only {len(rated)} rated pairs; need at least 8 to say anything")
        return

    vocab = build_vocab(args.vocab)
    bigrams = (BigramModel.from_file(str(args.bigrams), vocab=vocab)
               if args.bigrams.exists() else None)

    # x is the feature difference for the preferred minus the rejected member.
    data = []
    for p in rated:
        fa = features(p["a"]["words"], bigrams)
        fb = features(p["b"]["words"], bigrams)
        win, lose = (fa, fb) if p["prefer"] == "a" else (fb, fa)
        data.append([w - l for w, l in zip(win, lose)])

    rng = random.Random(args.seed)
    rng.shuffle(data)
    cut = max(4, int(0.75 * len(data)))
    train, test = data[:cut], data[cut:]

    theta = [0.0] * len(FEATURES)
    for _ in range(args.epochs):
        grad = [0.0] * len(theta)
        for x in train:
            p = _sigmoid(sum(t * xi for t, xi in zip(theta, x)))
            for j, xi in enumerate(x):
                grad[j] += (1.0 - p) * xi
        theta = [t + args.lr * (g / len(train) - args.l2 * t)
                 for t, g in zip(theta, grad)]

    def accuracy(rows):
        if not rows:
            return None
        return statistics.mean(
            1.0 if sum(t * xi for t, xi in zip(theta, x)) > 0 else 0.0
            for x in rows)

    tr, te = accuracy(train), accuracy(test)
    print(f"rated pairs {len(rated)}  train acc {tr:.2f}  "
          f"test acc {te if te is None else f'{te:.2f}'}  (chance 0.50)")
    print("\ncorrection weights:")
    for name, t in sorted(zip(FEATURES, theta), key=lambda r: -abs(r[1])):
        print(f"  {name:18s} {t:+.4f}")

    if te is not None and te <= 0.55:
        print("\nHeld-out accuracy is at chance. These preferences do not carry "
              "signal beyond the judge; leave the correction off.")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"features": FEATURES, "weights": theta, "rated_pairs": len(rated),
         "train_accuracy": tr, "test_accuracy": te}, indent=2))
    print(f"\nsaved {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("ask", help="write a file of pairs to rate")
    a.add_argument("--samples", type=Path, default=Path("runs/judge_samples.json"))
    a.add_argument("--out", type=Path, default=Path("runs/preferences.json"))
    a.add_argument("--n", type=int, default=30)
    a.add_argument("--arm", default="zipf",
                   help="which collection arm to draw from; 'any' to allow all")
    a.add_argument("--max-score-gap", type=float, default=0.05,
                   help="only pair texts GPT-2 scores about equally")
    a.add_argument("--max-overlap", type=float, default=0.5,
                   help="max Jaccard overlap of the two word sets")
    a.add_argument("--seed", type=int, default=0)
    a.set_defaults(func=ask)

    f = sub.add_parser("fit", help="fit a correction from the rated pairs")
    f.add_argument("--pairs", type=Path, default=Path("runs/preferences.json"))
    f.add_argument("--out", type=Path, default=Path("runs/preference_model.json"))
    f.add_argument("--bigrams", type=Path, default=Path("data/count_2w.txt"))
    f.add_argument("--vocab", type=int, default=30000)
    f.add_argument("--epochs", type=int, default=2000)
    f.add_argument("--lr", type=float, default=0.5)
    f.add_argument("--l2", type=float, default=0.05)
    f.add_argument("--seed", type=int, default=0)
    f.set_defaults(func=fit)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
