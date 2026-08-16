"""Measure how well each bank unit survives being mirrored, and store it.

Run after editing `data/word_banks.json`:

    PYTHONPATH=. python3 training/measure_reversal.py

Loads GPT-2, scores every three-word unit forward and reversed, and writes the
difference back into each bank under "reversal". The score is mean token
logprob per alphabetic character; a forward/reverse pair has identical letters,
so the normalisation cancels and the difference is a clean comparison.

Negative means the mirror reads worse than the original — the second half of
every paragraph containing that unit is grammatical but false.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

BANKS_PATH = Path("data/word_banks.json")


def gap_key(unit: str) -> str:
    """How a unit is keyed in the stored table.

    Selection reads units straight out of the bank, where they are capitalised;
    the table is written once from the same source. Normalising both ends means
    a re-cased bank edit does not silently orphan its measurement.
    """
    return " ".join(unit.lower().split())


def measure(banks: dict, scorer) -> dict:
    """Return {bank name: {unit key: gap}} for every three-word unit."""
    from llm_palindrome.reversal import reverse_unit

    texts: list[str] = []
    index: list[tuple[str, str]] = []
    for name, bank in banks.items():
        for unit in bank["outer"]:
            if len(unit.split()) != 3:
                continue
            texts.append(unit.capitalize() + ".")
            texts.append(reverse_unit(unit).capitalize() + ".")
            index.append((name, gap_key(unit)))

    scores = scorer.score_texts(texts)
    out: dict[str, dict[str, float]] = {name: {} for name in banks}
    for i, (name, key) in enumerate(index):
        out[name][key] = round(scores[2 * i + 1] - scores[2 * i], 4)
    return out


def main() -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from llm_palindrome.lm_scoring import GPT2Scorer

    banks = json.loads(BANKS_PATH.read_text())
    gaps = measure(banks, GPT2Scorer())
    for name, table in gaps.items():
        banks[name]["reversal"] = table
        worst = sorted(table.items(), key=lambda kv: kv[1])[:3]
        mean = sum(table.values()) / max(1, len(table))
        print(f"{name:<12} n={len(table):<3} mean={mean:+.4f}  worst: "
              + ", ".join(f"{k} ({v:+.2f})" for k, v in worst))

    BANKS_PATH.write_text(json.dumps(banks, indent=2) + "\n")
    print(f"wrote {BANKS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
