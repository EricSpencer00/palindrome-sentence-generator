"""Freeze the two long-form examples discussed in the paper."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from server.v2 import letter_paragraph

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ("data/novel_pairs.json", "data/mirror_units.json", "data/centres.json",
          "data/known_palindromes.json")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/long-form-examples-2026-09-11/examples.json")
    args = parser.parse_args()
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "server.v2.letter_paragraph",
        "inputs_sha256": {name: digest(ROOT / name) for name in INPUTS},
        "examples": {
            "generated": letter_paragraph(sentences=9, min_words=100, source="novel"),
            "catalogue": letter_paragraph(sentences=9, min_words=100, source="catalogue"),
        },
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
