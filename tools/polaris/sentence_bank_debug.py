"""Harvest diverse short mirror-pairs for hierarchical sentence composition."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from llm_palindrome.bigram import BigramModel
from llm_palindrome.hierarchy import CONNECTIVES
from llm_palindrome.overhang import DebtIndex, OverhangAware
from llm_palindrome.search import WordTries, beam_search
from llm_palindrome.semantic import RankOrderScorer
from llm_palindrome.validator import is_palindrome, normalize
from tools.polaris.shard_yield import load_brown, sentence_like


def rank_and_size(default_size: int) -> tuple[int, int]:
    for rank_name, size_name in (("PALS_RANKID", "PALS_LOCAL_SIZE"),
                                 ("PMI_RANK", "PMI_SIZE"),
                                 ("SLURM_PROCID", "SLURM_NTASKS")):
        if rank_name in os.environ:
            return int(os.environ[rank_name]), int(os.environ.get(size_name) or default_size)
    return 0, default_size


def split_pair(words: list[str]):
    half = sum(len(word) for word in words) / 2
    run = 0
    for index, word in enumerate(words):
        run += len(word)
        if run == half:
            left, right = words[:index + 1], words[index + 1:]
            return (left, right) if left and right else None
        if run > half:
            return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vocab", type=int, default=30000)
    ap.add_argument("--candidate-limit", type=int, default=200)
    ap.add_argument("--opening-pool", type=int, default=2048)
    ap.add_argument("--beam", type=int, default=64)
    ap.add_argument("--per-parent", type=int, default=8)
    ap.add_argument("--min-letters", type=int, default=28)
    ap.add_argument("--max-steps", type=int, default=180)
    ap.add_argument("--seeds-per-rank", type=int, default=32)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    rank, size = rank_and_size(args.shards)
    words = open(f"{HERE}/payload/vocab30k.txt").read().split()[:args.vocab]
    tries = WordTries(words)
    bigrams = BigramModel.from_file(f"{HERE}/payload/count_2w.txt", vocab=words)
    table, shapes, _ = load_brown(f"{HERE}/payload/brown.json.gz")
    opening_pool = tries.left_candidates("", args.opening_pool)
    rank_openings = opening_pool[rank::size]
    used_openings = set()
    seen = set()
    rows = []
    t0 = time.time()
    for offset in range(args.seeds_per_rank):
        seed = rank * args.seeds_per_rank + offset
        scorer = OverhangAware(
            RankOrderScorer(words, bigrams, order_weight=0.25),
            DebtIndex(tries, limit=96), debt_weight=2.0, dead_penalty=30.0)
        units = beam_search(
            tries, scorer, min_letters=args.min_letters,
            beam_width=args.beam, per_parent=args.per_parent,
            candidate_limit=args.candidate_limit, max_steps=args.max_steps,
            seed=seed, opening_words=set(rank_openings) - used_openings,
            max_word_uses=2)
        text = " ".join(units)
        if units:
            assert is_palindrome(text)
            used_openings.add(units[0])
        got = split_pair(units) if units else None
        accepted = False
        left = right = []
        if got:
            left, right = got
            key = (normalize(" ".join(left)), normalize(" ".join(right)))
            content = [word for word in left + right if word not in CONNECTIVES]
            accepted = (key not in seen
                        and sentence_like(left, table, shapes)
                        and sentence_like(right, table, shapes)
                        and max((content.count(word) for word in set(content)), default=0) <= 2)
            if accepted:
                seen.add(key)
        rows.append({"rank": rank, "seed": seed, "closed": bool(units),
                     "letters": len(normalize(text)), "opening": units[0] if units else None,
                     "text": text, "split": bool(got), "accepted": accepted,
                     "left": " ".join(left), "right": " ".join(right)})

    os.makedirs(args.out_dir, exist_ok=True)
    summary = {"rank": rank, "shards": size, "runs": len(rows),
               "closed": sum(row["closed"] for row in rows),
               "split": sum(row["split"] for row in rows),
               "accepted": sum(row["accepted"] for row in rows),
               "seconds": round(time.time() - t0, 2), "rows": rows}
    with open(f"{args.out_dir}/summary_r{rank:04d}.json", "w") as fh:
        json.dump(summary, fh)
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}),
          flush=True)


if __name__ == "__main__":
    main()
