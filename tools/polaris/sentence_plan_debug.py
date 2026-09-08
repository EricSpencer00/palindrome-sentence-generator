"""Benchmark and harvest exhaustive palindrome pairs with grammar-plan pruning."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from llm_palindrome.exhaustive import enumerate_palindromes
from llm_palindrome.bigram import BigramModel
from llm_palindrome.pairs import acceptable_pair, split_at_mirror
from llm_palindrome.search import WordTries
from llm_palindrome.sentence_plan import SentencePlan
from llm_palindrome.validator import is_palindrome, normalize
from tools.polaris.shard_yield import load_brown, rank_and_size


def run_arm(name, tries, plan, bigrams, args, rank, size):
    stats = {}
    rows, seen = [], set()
    started = time.time()
    state_gate = plan.state_possible if name != "terminal" else None
    allow_join = bigrams.observed if name.startswith("planned_join") else None
    join_slack = 1 if name == "planned_join1" else 0
    for units in enumerate_palindromes(
            tries, min_letters=args.min_letters, max_letters=args.max_letters,
            max_overhang=args.max_overhang, max_units=args.max_units,
            shard=rank, shards=size, node_budget=args.node_budget,
            deadline=started + args.seconds_per_arm, shuffle_seed=rank,
            allow_state=state_gate, allow_join=allow_join,
            join_slack=join_slack, stats=stats):
        split = split_at_mirror(units)
        if split is None:
            continue
        left, right = split
        if not acceptable_pair(left, right, min_words=args.min_words):
            continue
        if not (plan.complete(left) and plan.complete(right)):
            continue
        key = (" ".join(left), " ".join(right))
        if key in seen:
            continue
        seen.add(key)
        text = " ".join(units)
        assert is_palindrome(text), text
        rows.append({"left": key[0], "right": key[1],
                     "letters": len(normalize(text)), "text": text})
        if len(rows) >= args.max_hits:
            break
    return {"arm": name, "rank": rank, "nodes": stats.get("nodes", 0),
            "closures": stats.get("yielded", 0),
            "state_pruned": stats.get("state_pruned", 0),
            "hits": len(rows), "seconds": round(time.time() - started, 3),
            "rows": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vocab", type=int, default=30000)
    ap.add_argument("--min-letters", type=int, default=20)
    ap.add_argument("--max-letters", type=int, default=44)
    ap.add_argument("--min-words", type=int, default=3)
    ap.add_argument("--max-units", type=int, default=18)
    ap.add_argument("--max-overhang", type=int, default=16)
    ap.add_argument("--node-budget", type=int, default=20_000_000)
    ap.add_argument("--seconds-per-arm", type=float, default=600)
    ap.add_argument("--max-hits", type=int, default=10000)
    ap.add_argument("--arms", default="terminal,planned")
    ap.add_argument("--bigram-min-count", type=int, default=1)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    rank, size = rank_and_size(args.shards)
    table, shapes, _ = load_brown(f"{HERE}/payload/brown.json.gz")
    raw = open(f"{HERE}/payload/vocab30k.txt").read().split()[:args.vocab]
    # Unknown words can never complete a Brown plan. Removing them gives both
    # arms the same honest search space and prevents a misleading speedup.
    words = [word for word in raw if word in table]
    tries = WordTries(words)
    plan = SentencePlan(table, shapes, args.min_words, args.max_units // 2)
    requested = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    bigram_path = f"{HERE}/payload/count_2w.txt"
    if not os.path.exists(bigram_path):
        bigram_path = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                                   "data", "count_2w.txt")
    bigrams = (BigramModel.from_file(bigram_path,
                                     vocab=words,
                                     min_count=args.bigram_min_count)
               if any(arm.startswith("planned_join") for arm in requested)
               else None)
    summaries = [run_arm(arm, tries, plan, bigrams, args, rank, size)
                 for arm in requested]
    os.makedirs(args.out_dir, exist_ok=True)
    path = f"{args.out_dir}/summary_r{rank:04d}.json"
    with open(path, "w") as fh:
        json.dump({"rank": rank, "shards": size, "vocab": len(words),
                   "arms": summaries}, fh)
    print(json.dumps({"rank": rank, "vocab": len(words),
                      "arms": [{k: v for k, v in arm.items() if k != "rows"}
                               for arm in summaries]}), flush=True)


if __name__ == "__main__":
    main()
