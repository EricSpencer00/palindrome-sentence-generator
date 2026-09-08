"""Matched semantic-search screen for laptop or Polaris debug nodes.

Every arm uses the same vocabulary, seeds, beam, parent quota, and debt term.
Only the weight on bidirectional bigram order gain changes. Exact output text
is retained so an unchanged language model can judge the pooled candidates
after the inexpensive cluster search has finished.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from llm_palindrome.bigram import BigramModel
from llm_palindrome.overhang import DebtIndex, OverhangAware
from llm_palindrome.search import WordTries, beam_search
from llm_palindrome.semantic import RankOrderScorer
from llm_palindrome.validator import is_palindrome, normalize


def rank_and_size(default_size: int) -> tuple[int, int]:
    for rank_name, size_name in (("PALS_RANKID", "PALS_LOCAL_SIZE"),
                                 ("PMI_RANK", "PMI_SIZE"),
                                 ("SLURM_PROCID", "SLURM_NTASKS")):
        if rank_name in os.environ:
            return int(os.environ[rank_name]), int(os.environ.get(size_name) or default_size)
    return 0, default_size


def text_metrics(words: list[str], bigrams: BigramModel) -> dict:
    pairs = list(zip(words, words[1:]))
    gains = [bigrams.forward_order_gain(a, b) for a, b in pairs]
    return {
        "distinct_ratio": round(len(set(words)) / max(1, len(words)), 4),
        "attested_pair_rate": round(
            sum(bigrams.observed(a, b) for a, b in pairs) / max(1, len(pairs)), 4),
        "mean_order_gain": round(statistics.mean(gains), 4) if gains else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vocab-file", default=f"{HERE}/payload/vocab30k.txt")
    ap.add_argument("--bigram-file", default=f"{HERE}/payload/count_2w.txt")
    ap.add_argument("--vocab", type=int, default=30000)
    ap.add_argument("--weights", default="0,0.25,0.5,1,2")
    ap.add_argument("--candidate-limit", type=int, default=200)
    ap.add_argument("--beam", type=int, default=48)
    ap.add_argument("--per-parent", type=int, default=8)
    ap.add_argument("--min-letters", type=int, default=80)
    ap.add_argument("--max-steps", type=int, default=240)
    ap.add_argument("--seeds-per-rank", type=int, default=4)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    rank, size = rank_and_size(args.shards)
    words = open(args.vocab_file).read().split()[:args.vocab]
    tries = WordTries(words)
    bigrams = BigramModel.from_file(args.bigram_file, vocab=words)
    weights = [float(value) for value in args.weights.split(",")]
    rows = []
    t0 = time.time()
    for weight in weights:
        base = RankOrderScorer(words, bigrams, order_weight=weight)
        scorer = OverhangAware(base, DebtIndex(tries, limit=96),
                               debt_weight=2.0, dead_penalty=30.0)
        for offset in range(args.seeds_per_rank):
            seed = rank * args.seeds_per_rank + offset
            units = beam_search(
                tries, scorer, min_letters=args.min_letters,
                beam_width=args.beam, per_parent=args.per_parent,
                candidate_limit=args.candidate_limit, max_steps=args.max_steps,
                seed=seed)
            text = " ".join(units)
            valid = bool(units) and is_palindrome(text)
            if units:
                assert valid, text
            row = {"rank": rank, "seed": seed, "weight": weight,
                   "closed": bool(units), "valid": valid,
                   "letters": len(normalize(text)), "words": len(units),
                   "opening": units[0] if units else None, "text": text}
            if units:
                row.update(text_metrics(units, bigrams))
            rows.append(row)

    os.makedirs(args.out_dir, exist_ok=True)
    summary = {"rank": rank, "shards": size, "vocab": args.vocab,
               "weights": weights, "runs": len(rows),
               "closed": sum(row["closed"] for row in rows),
               "seconds": round(time.time() - t0, 2), "rows": rows}
    path = f"{args.out_dir}/summary_r{rank:04d}.json"
    with open(path, "w") as fh:
        json.dump(summary, fh)
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}),
          flush=True)


if __name__ == "__main__":
    main()
