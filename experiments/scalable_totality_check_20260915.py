"""Evaluate the O(N) exact-length totality constructor.

The rows are deliberately marked non-reader evidence: this run evaluates the
arbitrary-size contract, not English quality.  Strict lexical closures remain
the separate path used for candidate promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.scalable import construct_total


def independent_ascii_tape(text: str) -> str:
    """Verifier deliberately separate from the constructor normalizer."""
    return "".join(
        character.casefold() for character in text
        if ("A" <= character <= "Z") or ("a" <= character <= "z")
    )


def run(targets: list[int]) -> dict:
    rows = []
    for target in targets:
        row = construct_total(int(target))
        tape = independent_ascii_tape(row["text"])
        rows.append({
            "target": target,
            "status": row["status"],
            "letters": len(tape),
            "exact": tape == tape[::-1] and len(tape) == target,
            "independent_verifier": "explicit_ascii_scan_v1",
            "reader_candidate": row["reader_candidate"],
            "fallback": row["fallback"],
            "admission": row["admission"],
        })
    return {
        "status": "complete_scalable_totality_check",
        "targets": targets,
        "rows": rows,
        "all_exact": all(row["exact"] for row in rows),
        "all_non_reader": all(not row["reader_candidate"] for row in rows),
        "algorithm": "construct_total: direct mirrored one-letter tiling, O(N) output work",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reader_gate": "This totality witness is never eligible for reader evaluation or API promotion.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--targets", type=int, nargs="+",
                        default=[1, 2, 31, 38, 39, 511, 1023, 10_001, 100_001])
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite output: {args.out}")
    result = run(args.targets)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"targets": args.targets, "all_exact": result["all_exact"]}, indent=2))


if __name__ == "__main__":
    main()
