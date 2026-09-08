"""Cluster smoke/closure benchmark for the corrected beam search.

This is intentionally not another yield sweep.  Each rank verifies that a
limited candidate menu still includes ordinary-length words, then runs several
independent exact-palindrome searches with a rank-based frequency proxy and a
repayable-overhang term.  It needs only the frozen vocabulary and the standard
library, so it is safe to stage to a fresh Polaris node.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from llm_palindrome.overhang import DebtIndex, OverhangAware
from llm_palindrome.search import WordTries, unit_letters
from llm_palindrome.search import beam_search
from llm_palindrome.validator import is_palindrome, normalize


def rank_and_size(default_size: int) -> tuple[int, int]:
    for rank_name, size_name in (("PALS_RANKID", "PALS_LOCAL_SIZE"),
                                 ("PMI_RANK", "PMI_SIZE"),
                                 ("SLURM_PROCID", "SLURM_NTASKS")):
        if rank_name in os.environ:
            return int(os.environ[rank_name]), int(os.environ.get(size_name) or default_size)
    return 0, default_size


class RankScorer:
    """A dependency-free stand-in for frequency ranking on the frozen vocab."""

    def __init__(self, words: list[str]):
        self.rank = {word: index for index, word in enumerate(words)}

    def word_delta(self, left, right, placement, word, growth) -> float:
        uses = left.count(word) + right.count(word) - 1
        # Input order is the frequency prior.  The modest length term prevents
        # the structural scorer from treating a word menu as a filler menu.
        return 8.0 - math.log1p(self.rank[word]) + 0.20 * len(unit_letters(word)) - 2.0 * uses


def menu_stats(words: list[str]) -> dict:
    lengths = [len(unit_letters(word)) for word in words]
    return {"n": len(words), "mean_letters": round(sum(lengths) / len(lengths), 3),
            "max_letters": max(lengths), "at_least_5": sum(n >= 5 for n in lengths),
            "sample": words[:12]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vocab", type=int, default=30000)
    ap.add_argument("--candidate-limit", type=int, default=200)
    ap.add_argument("--beam", type=int, default=48)
    ap.add_argument("--per-parent", type=int, default=8)
    ap.add_argument("--min-letters", type=int, default=80)
    ap.add_argument("--max-steps", type=int, default=240)
    ap.add_argument("--seeds-per-rank", type=int, default=8)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    rank, size = rank_and_size(args.shards)
    words = open(f"{HERE}/payload/vocab30k.txt").read().split()[:args.vocab]
    tries = WordTries(words)
    menu = tries.left_candidates("", args.candidate_limit)
    stats = menu_stats(menu)
    # This is the regression the job exists to catch: the former BFS menu had
    # max_letters=3 and at_least_5=0 at this exact call site.
    assert stats["at_least_5"] > 0 and stats["max_letters"] >= 5, stats

    debt = DebtIndex(tries, limit=96)
    scorer = OverhangAware(RankScorer(words), debt, debt_weight=2.0, dead_penalty=30.0)
    rows = []
    t0 = time.time()
    for offset in range(args.seeds_per_rank):
        seed = rank * args.seeds_per_rank + offset
        units = beam_search(tries, scorer, min_letters=args.min_letters,
                            beam_width=args.beam, per_parent=args.per_parent,
                            candidate_limit=args.candidate_limit,
                            max_steps=args.max_steps, seed=seed)
        text = " ".join(units)
        valid = bool(units) and is_palindrome(text)
        if units:
            assert valid, text
        rows.append({"seed": seed, "closed": bool(units), "valid": valid,
                     "letters": len(normalize(text)),
                     "opening": units[0] if units else None,
                     "words": len(units)})

    os.makedirs(args.out_dir, exist_ok=True)
    summary = {"rank": rank, "shards": size, "vocab": args.vocab,
               "candidate_limit": args.candidate_limit, "menu": stats,
               "closed": sum(row["closed"] for row in rows), "runs": len(rows),
               "distinct_openings": len({row["opening"] for row in rows if row["opening"]}),
               "seconds": round(time.time() - t0, 2), "rows": rows}
    path = f"{args.out_dir}/summary_r{rank:04d}.json"
    with open(path, "w") as fh:
        json.dump(summary, fh)
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}), flush=True)


if __name__ == "__main__":
    main()
