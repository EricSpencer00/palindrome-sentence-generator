"""Exhaustive sentence-plan walk with clause-state sibling ordering."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_palindrome.clause_ngram import ClauseNgramScorer
from llm_palindrome.exhaustive import enumerate_palindromes
from llm_palindrome.pairs import acceptable_pair, split_at_mirror
from llm_palindrome.search import WordTries
from llm_palindrome.sentence_plan import SentencePlan
from llm_palindrome.syntax import brown_tables
from llm_palindrome.validator import is_palindrome


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vocab", type=int, default=3000)
    ap.add_argument("--orders", default="2,4")
    ap.add_argument("--node-budget", type=int, default=250000)
    ap.add_argument("--min-letters", type=int, default=20)
    ap.add_argument("--max-letters", type=int, default=44)
    ap.add_argument("--out", type=Path,
                    default=Path("runs/joint_clause_walk.json"))
    args = ap.parse_args()

    from nltk.corpus import brown
    corpus = [[word.lower() for word in sentence if word.isalpha()]
              for sentence in brown.sents()]
    table, shapes, _ = brown_tables(3, 9)
    raw = Path("tools/polaris/payload/vocab30k.txt").read_text().split()
    words = [word for word in raw[:args.vocab] if word in table]
    tries = WordTries(words)
    plan = SentencePlan(table, shapes, 3, 9)
    arms = []
    for order in (int(x) for x in args.orders.split(",")):
        scorer = ClauseNgramScorer(corpus, order=order)
        stats, rows, seen = {}, [], set()
        started = time.time()
        for units in enumerate_palindromes(
                tries, min_letters=args.min_letters, max_letters=args.max_letters,
                max_units=18, max_overhang=16, node_budget=args.node_budget,
                allow_state=plan.state_possible, scorer=scorer, stats=stats):
            split = split_at_mirror(units)
            if split is None:
                continue
            left, right = split
            if not (acceptable_pair(left, right, 3)
                    and plan.complete(left) and plan.complete(right)):
                continue
            key = (" ".join(left), " ".join(right))
            if key in seen:
                continue
            seen.add(key)
            assert is_palindrome(" ".join(units))
            rows.append({"left": key[0], "right": key[1]})
        arm = {"order": order, "nodes": stats.get("nodes", 0),
               "pruned": stats.get("state_pruned", 0), "hits": len(rows),
               "seconds": round(time.time() - started, 2), "rows": rows}
        arms.append(arm)
        print(json.dumps({k: v for k, v in arm.items() if k != "rows"}), flush=True)
        for row in rows[:12]:
            print(f"  {row['left']} | {row['right']}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"config": vars(args) | {"out": str(args.out)},
                                    "arms": arms}, indent=2) + "\n")


if __name__ == "__main__":
    main()
