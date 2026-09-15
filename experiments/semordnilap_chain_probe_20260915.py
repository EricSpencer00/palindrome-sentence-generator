"""Bounded probe of independently rendered semordnilap chains.

The chain constructor pairs each authored left word with an ordinary reverse
pair on the right, then independently renders both clause halves.  This is a
diagnostic probe: exactness is checked from the rendered text and admission is
the shared fail-closed gate.  It intentionally does not relax word-order
symmetry or catalogue protections.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

PAIRS = {
    "draw": "ward", "drawer": "reward", "diaper": "repaid", "live": "evil",
    "stressed": "desserts", "deliver": "reviled", "gateman": "nametag",
    "dog": "god", "part": "trap", "stop": "pots", "flow": "wolf",
    "star": "rats", "smart": "trams", "saw": "was", "raw": "war",
    "snug": "guns", "loop": "pool", "deer": "reed", "emit": "time",
}

# Complete, independently grammatical clause shells.  The selected lexical
# slots are filled with reverse-paired words; the reverse side is composed as
# a separate clause rather than copied from a catalogue palindrome.
LEFT = (
    ("I", "draw"), ("I", "live"), ("we", "saw"), ("we", "stop"),
    ("they", "deliver"), ("a", "drawer"), ("the", "smart"),
    ("we", "loop"), ("I", "emit"), ("the", "deer"),
)
RIGHT_SHELL = {
    ("I", "draw"): ("ward", "I"), ("I", "live"): ("evil", "I"),
    ("we", "saw"): ("was", "we"), ("we", "stop"): ("pots", "we"),
    ("they", "deliver"): ("reviled", "they"),
    ("a", "drawer"): ("reward", "a"), ("the", "smart"): ("trams", "the"),
    ("we", "loop"): ("pool", "we"), ("I", "emit"): ("time", "I"),
    ("the", "deer"): ("reed", "the"),
}

def render(words: tuple[str, ...]) -> str:
    return " ".join(words).capitalize() + "."

def main() -> None:
    rows = []
    exact = admitted = 0
    for width in (1, 2, 3):
        for combo in itertools.combinations(LEFT, width):
            left = tuple(word for slot in combo for word in slot)
            # Right clause is authored from the lexical reverse map, in reverse
            # slot order, while retaining its own grammatical shell words.
            right = tuple(word for slot in reversed(combo) for word in RIGHT_SHELL[slot])
            text = render(left) + " " + render(right).lower()
            tape = normalize_letters(text)
            independent = tape == tape[::-1]
            checks = mechanical_admission_checks(text, min_letters=30, max_letters=100)
            row = {"width": width, "rendered": text, "tokens": list(tokenize(text)),
                   "letters": len(tape), "independent_exact": independent,
                   "admitted": all(checks.values()),
                   "failed_checks": [k for k, v in checks.items() if not v]}
            rows.append(row)
            exact += independent
            admitted += independent and row["admitted"]
    out = {"probe": "semordnilap_chain", "candidate_count": len(rows),
           "exact_count": exact, "admitted_count": admitted,
           "target_exact_gt": 38, "target_met": exact > 38,
           "target_admitted_gt": 38, "admitted_target_met": admitted > 38,
           "common_failure_counts": {}, "rows": rows}
    failures = out["common_failure_counts"]
    for row in rows:
        for key in row["failed_checks"]:
            failures[key] = failures.get(key, 0) + 1
    path = ROOT / "runs" / "semordnilap-chain-probe-2026-09-15.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("candidate_count", "exact_count", "admitted_count", "target_met", "admitted_target_met", "common_failure_counts")}, indent=2))

if __name__ == "__main__":
    main()
