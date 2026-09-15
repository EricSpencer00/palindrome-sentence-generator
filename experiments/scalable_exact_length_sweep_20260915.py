"""Replay the length-indexed exact constructor over an arbitrary target sweep.

The strict rows are lexical search evidence.  The explicit fallback rows only
test that the target-length engine is total; they are never readability or
reader-study candidates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from wordfreq import top_n_list

from llm_palindrome.scalable import construct_many


def run(targets: list[int], vocabulary_size: int = 180) -> dict:
    vocabulary = [
        word for word in top_n_list("en", vocabulary_size)
        if word.isascii() and word.isalpha()
    ]
    rows = construct_many(
        targets, vocabulary, max_nodes=10_000, candidate_limit=128,
        fallback_one_letters=True,
    )
    return {
        "status": "exact_length_sweep_complete",
        "targets": targets,
        "vocabulary_size": len(vocabulary),
        "vocabulary_sha256": hashlib.sha256("\n".join(vocabulary).encode()).hexdigest(),
        "rows": rows,
        "reader_gate": (
            "Fallback rows are construction-core diagnostics only. Any strict lexical row "
            "must still pass the shared mechanical gate and randomized blinded human readers."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--targets", type=int, nargs="+", default=[31, 47, 63, 95])
    ap.add_argument("--vocabulary-size", type=int, default=180)
    args = ap.parse_args()
    if args.out.exists():
        ap.error(f"refusing to overwrite output: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = run(args.targets, args.vocabulary_size)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"targets": args.targets,
                      "statuses": [row["status"] for row in result["rows"]]}, indent=2))


if __name__ == "__main__":
    main()

