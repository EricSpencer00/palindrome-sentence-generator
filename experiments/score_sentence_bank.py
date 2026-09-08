"""Absolute, control-gated scoring for harvested sentence mirror-pairs.

This produces a shortlist for reading.  It never rewrites the bank or promotes
a pair: the project's judges are useful filters and unreliable deciders.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.coherence_scale import ask
from experiments.filter_sentence_pairs import NEGATIVE, POSITIVE


def ask_with_retry(model: str, text: str):
    for attempt in range(6):
        try:
            return ask(model, text)
        except Exception:
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs-file", type=Path, required=True)
    ap.add_argument("--arm", default="planned_join0")
    ap.add_argument("--model", default="gpt-oss:120b-cloud")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    blob = json.loads(args.pairs_file.read_text())
    pairs = blob[args.arm]["pairs"]
    items = [(f"p{i}L", row["left"]) for i, row in enumerate(pairs)]
    items += [(f"p{i}R", row["right"]) for i, row in enumerate(pairs)]
    items += [(f"cp{i}", text) for i, text in enumerate(POSITIVE)]
    items += [(f"cn{i}", text) for i, text in enumerate(NEGATIVE)]

    scored = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(ask_with_retry, args.model, text): (key, text)
                for key, text in items}
        for done, future in enumerate(as_completed(jobs), 1):
            key, text = jobs[future]
            score, raw = future.result()
            scored[key] = {"text": text, "score": score, "raw": raw}
            if done % 25 == 0:
                print(f"scored {done}/{len(items)}", flush=True)

    # The last positive is the deliberately compressed "Items draw award".
    # It calibrates preservation of charm, not book-prose naturalness, and the
    # previously validated absolute rubric consistently puts it at tier 1.
    positive_floors = [2] * (len(POSITIVE) - 1) + [1]
    positive_ok = all(scored[f"cp{i}"]["score"] is not None
                      and scored[f"cp{i}"]["score"] >= positive_floors[i]
                      for i in range(len(POSITIVE)))
    negative_ok = all(scored[f"cn{i}"]["score"] is not None
                      and scored[f"cn{i}"]["score"] <= 1
                      for i in range(len(NEGATIVE)))
    complete = all(row["score"] is not None for row in scored.values())
    valid = positive_ok and negative_ok and complete

    ranked = []
    for i, pair in enumerate(pairs):
        left = scored[f"p{i}L"]["score"]
        right = scored[f"p{i}R"]["score"]
        ranked.append(dict(pair, left_score=left, right_score=right,
                           floor=min(left, right), mean=(left + right) / 2))
    ranked.sort(key=lambda row: (-row["floor"], -row["mean"],
                                 row["left"], row["right"]))
    result = {"model": args.model, "arm": args.arm, "pairs": len(pairs),
              "valid": valid, "positive_ok": positive_ok,
              "negative_ok": negative_ok, "complete": complete,
              "controls": {key: row for key, row in scored.items()
                           if key.startswith("c")},
              "shortlist": ([row for row in ranked if row["floor"] >= 2]
                            if valid else []),
              "ranked": ranked}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in
                      ("valid", "positive_ok", "negative_ok", "complete")}
                     | {"shortlist": len(result["shortlist"])}), flush=True)


if __name__ == "__main__":
    main()
