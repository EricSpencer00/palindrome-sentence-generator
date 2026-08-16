"""Cut a hunt's output down to something a person can actually read.

`pair_hunt.py` returns thousands of letter-valid pairs ranked by GPT-2. That
ranking is a filter and not a verdict — the project's record on proxies is four
disagreements with blind judging and no agreements — so the last step before a
pair is allowed into `data/novel_pairs.json` is a human reading it.

The shortlist is ordered by a second, independent signal: how many of the joins
in each half are word pairs English has actually been seen to make. A half
whose every join is attested is not guaranteed to read, but a half with none
attested reliably does not, and the two signals disagree often enough that
ranking by both surfaces different material than ranking by either.

    python experiments/pair_shortlist.py --in runs/pair_hunt.json --top 200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_palindrome.mining import attested_bigrams


def attested_fraction(words, attested) -> float:
    """Share of this half's joins that are attested. One word has no joins."""
    joins = list(zip(words, words[1:]))
    if not joins:
        return 0.0
    return sum((a, b) in attested for a, b in joins) / len(joins)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", default="runs/pair_hunt.json")
    ap.add_argument("--bigrams", default="data/count_2w.txt")
    ap.add_argument("--top", type=int, default=200)
    ap.add_argument("--min-attested", type=float, default=0.0,
                    help="floor on the WEAKER half's attested-join share")
    ap.add_argument("--out", default="runs/pair_shortlist.json")
    args = ap.parse_args()

    rows = json.loads(Path(args.src).read_text())["results"]
    attested = attested_bigrams(args.bigrams)

    for row in rows:
        left = attested_fraction(row["left"], attested)
        right = attested_fraction(row["right"], attested)
        row["attested"] = round(min(left, right), 3)
        row["attestedMean"] = round((left + right) / 2, 3)

    kept = [r for r in rows if r["attested"] >= args.min_attested]
    kept.sort(key=lambda r: (-r["attested"], -r.get("score", 0.0)))
    kept = kept[:args.top]

    Path(args.out).write_text(json.dumps(kept, indent=1))
    print(f"{len(rows)} pairs -> {len(kept)} shortlisted -> {args.out}\n")
    for row in kept:
        print(f"  {row['attested']:.2f} {row.get('score', 0):7.3f}  "
              f"{' '.join(row['left'])} || {' '.join(row['right'])}")


if __name__ == "__main__":
    main()
