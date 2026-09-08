"""Jointly decode two mirrored clauses with directional multiword state."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_palindrome.centerout import centerout_search
from llm_palindrome.clause_ngram import ClauseNgramScorer
from llm_palindrome.pairs import acceptable_pair, split_at_mirror
from llm_palindrome.search import WordTries
from llm_palindrome.sentence_plan import SentencePlan
from llm_palindrome.syntax import brown_tables
from llm_palindrome.validator import is_palindrome, normalize


def brown_words():
    from nltk.corpus import brown
    return [[word.lower() for word in sentence if word.isalpha()]
            for sentence in brown.sents()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vocab", type=int, default=6000)
    ap.add_argument("--orders", default="2,4")
    ap.add_argument("--seeds", type=int, default=64)
    ap.add_argument("--beam", type=int, default=128)
    ap.add_argument("--candidate-limit", type=int, default=256)
    ap.add_argument("--min-letters", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--out", type=Path,
                    default=Path("runs/joint_clause_search.json"))
    args = ap.parse_args()

    corpus = brown_words()
    table, shapes, _ = brown_tables(3, 9)
    raw = Path("tools/polaris/payload/vocab30k.txt").read_text().split()
    words = [word for word in raw[:args.vocab] if word in table]
    tries = WordTries(words)
    plan = SentencePlan(table, shapes, min_words=3, max_words=9)
    arms = []
    for order in (int(value) for value in args.orders.split(",")):
        scorer = ClauseNgramScorer(corpus, order=order)
        rows, started = [], time.time()
        for seed in range(args.seeds):
            units = centerout_search(
                tries, scorer, min_letters=args.min_letters,
                beam_width=args.beam, candidate_limit=args.candidate_limit,
                per_parent=8, max_steps=args.max_steps, seed=seed,
                diversity=0.25, max_overhang=16,
                allow_state=plan.state_possible,
                allow_closed=lambda left, right:
                    plan.complete(left) and plan.complete(right))
            split = split_at_mirror(units)
            if split is None:
                continue
            left, right = split
            if not acceptable_pair(left, right, min_words=3):
                continue
            text = " ".join(units)
            assert is_palindrome(text)
            rows.append({"seed": seed, "left": " ".join(left),
                         "right": " ".join(right),
                         "letters": len(normalize(text)), "text": text})
        arms.append({"order": order, "attempts": args.seeds,
                     "closed": len(rows), "distinct": len({row["text"] for row in rows}),
                     "seconds": round(time.time() - started, 2), "rows": rows})
        print(f"order={order} closed={len(rows)}/{args.seeds} "
              f"distinct={arms[-1]['distinct']} seconds={arms[-1]['seconds']}",
              flush=True)
        for row in rows[:10]:
            print(f"  {row['left']} | {row['right']}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"config": vars(args) | {"out": str(args.out)},
                                    "arms": arms}, indent=2) + "\n")


if __name__ == "__main__":
    main()
