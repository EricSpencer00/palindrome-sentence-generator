"""Where does exhaustive enumeration stop being exhaustive?

The repository asserts a boundary near 30 letters and has never measured it.
The figure appears as 24 in one docstring, "at most 28" in `docs/training.md`
and "~30" in the paper, and the next band anyone searched was 40-60. The 31-39
band, which is exactly where the question lives, is unmeasured.

The operational test
--------------------
`enumerate_palindromes` takes a `node_budget`, so "exhaustive" is checkable
without trusting anything: run the same band at rising budgets and watch the
count of distinct palindromes. A walk that has seen the whole space returns the
same number however much more budget it is given. A walk that is sampling one
corner keeps climbing, roughly in proportion to the budget it is handed.

The reported quantity is therefore the SATURATION RATIO: the count at the
largest budget divided by the count at half that budget. A completed walk
scores 1.00. A walk still discovering at the same rate as its budget scores
near 2.00. Somewhere between those the band stops being enumerable, and that
crossing is the boundary the paper should be quoting.

The boundary is not a property of English
-----------------------------------------
It is a property of English AND the vocabulary, because branching at every
closure goes with the number of units that fit the overhang. The sweep
therefore runs several vocabulary sizes, and the answer is a curve rather than
a number. Reporting it as one number without its vocabulary is the mistake this
script exists to stop repeating.

Usage
-----
    python experiments/regime_boundary.py
    python experiments/regime_boundary.py --vocabs 400 1500 --budget 400000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_palindrome.exhaustive import enumerate_palindromes
from llm_palindrome.generate import build_vocab
from llm_palindrome.search import WordTries
from llm_palindrome.validator import is_palindrome, normalize

BANDS = [(12, 16), (17, 21), (22, 26), (27, 31), (32, 36), (37, 41), (42, 46)]


def walk(tries: WordTries, lo: int, hi: int, budget: int,
         seed: int = 0, cap: int = 400_000, max_units: int = 40) -> set[str]:
    """Distinct normalised palindromes in [lo, hi] letters within `budget`.

    Deduplicated by letters rather than by word sequence: two segmentations of
    one letter string are one palindrome for the purpose of asking how big the
    space is.

    `max_units` must be raised well above the module default of 12 or it, and
    not the letter band, is what the sweep measures. A 60-word vocabulary of
    mostly short words needs more than twelve units to reach forty letters, and
    at the default the 32-36 band returns 17 palindromes against 87,693 with
    the cap lifted. Every "the space is empty up there" reading of an earlier
    walk should be checked against this.
    """
    out: set[str] = set()
    for words in enumerate_palindromes(tries, max_letters=hi, min_letters=lo,
                                       node_budget=budget, max_units=max_units,
                                       shuffle_seed=seed):
        out.add(normalize(" ".join(words)))
        if len(out) >= cap:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vocabs", nargs="+", type=int, default=[300, 1000, 3000])
    ap.add_argument("--budget", type=int, default=300_000,
                    help="the larger of the two node budgets")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-units", type=int, default=40,
                    help="raise well above the module default of 12, or the "
                         "unit cap rather than the letter band is measured")
    ap.add_argument("--out", default="experiments/regime_boundary.json")
    args = ap.parse_args()

    full = build_vocab(30000)
    rows = []
    for v in args.vocabs:
        tries = WordTries(full[:v])
        print(f"\n=== vocabulary {v} words ===")
        print(f"{'band':>9} {'half budget':>12} {'full budget':>12} "
              f"{'ratio':>6} {'verdict':>12} {'secs':>6}")
        for lo, hi in BANDS:
            t0 = time.time()
            small = walk(tries, lo, hi, args.budget // 2, args.seed,
                         max_units=args.max_units)
            big = walk(tries, lo, hi, args.budget, args.seed,
                       max_units=args.max_units)
            secs = time.time() - t0
            if not small:
                ratio = float("nan")
                verdict = "empty"
            else:
                ratio = len(big) / len(small)
                # 1.00 means the extra budget found nothing new.
                verdict = ("exhaustive" if ratio < 1.02 else
                           "near" if ratio < 1.20 else "sampling")
            rows.append({"vocab": v, "lo": lo, "hi": hi,
                         "n_half": len(small), "n_full": len(big),
                         "ratio": ratio, "verdict": verdict, "seconds": secs})
            print(f"{lo:3d}-{hi:<5d} {len(small):12d} {len(big):12d} "
                  f"{ratio:6.2f} {verdict:>12} {secs:6.1f}")

    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nwrote {args.out}")

    print("\nlast band that completes, by vocabulary:")
    for v in args.vocabs:
        done = [r for r in rows if r["vocab"] == v and r["verdict"] == "exhaustive"
                and r["n_full"] > 0]
        if done:
            r = max(done, key=lambda r: r["hi"])
            print(f"  {v:5d} words: up to {r['hi']} letters "
                  f"({r['n_full']} palindromes)")
        else:
            print(f"  {v:5d} words: none of the bands completed")


if __name__ == "__main__":
    main()
