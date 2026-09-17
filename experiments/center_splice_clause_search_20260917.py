"""Bounded center-splice experiment.

The two clause inventories are authored independently.  A seam is live: the
search chooses a center token and asks whether the concatenation closes.  It
does not re-segment a stored tape, and it rejects the common semordnilap
shortcut (each right word being the reverse of a left word).
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
from pathlib import Path

LEFT = (
    "deliver stressed drawer",
    "quiet civic radar",
    "kind noon civic",
    "read level civic",
)
RIGHT = (
    "reward desserts reviled",
    "radar civic quiet",
    "civic noon kind",
    "civic level read",
)
CENTRES = ("level", "noon", "civic", "rotor")
WORD = re.compile(r"[a-z]+")


def letters(text: str) -> str:
    return "".join(c.lower() for c in text if c.isalpha())


def semordnilap_shortcut(left: str, right: str) -> bool:
    a, b = WORD.findall(left.lower()), WORD.findall(right.lower())
    return bool(a and len(a) == len(b) and all(x[::-1] == y for x, y in zip(a, b[::-1])))


def run() -> dict:
    tested = exact = rejected = []
    tested = []
    for left, right, centre in itertools.product(LEFT, RIGHT, CENTRES):
        text = f"{left}; {centre}; {right}."
        norm = letters(text)
        row = {"text": text, "letters": len(norm), "exact": norm == norm[::-1]}
        tested.append(row)
        if row["exact"]:
            exact.append(row) if isinstance(exact, list) else None
    admitted = [r for r in exact if 40 <= r["letters"] <= 100 and not semordnilap_shortcut(r["text"].split(";")[0], r["text"].split(";")[-1])]
    rejected = [r for r in exact if semordnilap_shortcut(r["text"].split(";")[0], r["text"].split(";")[-1])]
    payload = {
        "method": "independent_clause_center_splice",
        "bounds": {"left": len(LEFT), "right": len(RIGHT), "centres": len(CENTRES), "combinations": len(tested)},
        "exact_candidates": exact,
        "rejected_shortcut_candidates": rejected,
        "admitted": admitted,
        "sha256": hashlib.sha256(json.dumps(tested, sort_keys=True).encode()).hexdigest(),
    }
    out = Path("artifacts/center_splice_clause_search_20260917.json")
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


if __name__ == "__main__":
    result = run()
    print(json.dumps({k: result[k] for k in ("bounds", "exact_candidates", "admitted", "sha256")}, indent=2))
