"""Differentially verified midpoint-crossing product.

Earlier clause products treated the seam between two complete clauses as the
palindrome midpoint.  That is an unnecessary restriction: the midpoint may
fall inside a word, and the two independently chosen clause spans may have
different lengths.  This experiment validates a cursor product against a
full-tape oracle on tiny variable-length fixtures, then applies the same
state machine to the existing recursive prose grammar.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/midpoint-crossing-product-20260917.json"
ID = "midpoint-crossing-product-20260917"
SIG = "midpoint-crossing-cursors|variable-length-derivations|internal-word-center|oracle-differential|independent-audit"


def normalize(text: str) -> str:
    return "".join(c for c in text.casefold() if c.isascii() and c.isalpha())


def oracle(text: str) -> bool:
    letters = normalize(text)
    return bool(letters) and letters == letters[::-1]


def crossing_product(tokens: tuple[str, ...] | list[str]) -> dict:
    """Walk one rendered derivation from both ends, allowing any midpoint."""
    chars: list[tuple[int, int, str]] = []
    for token_index, token in enumerate(tokens):
        for char_index, char in enumerate(normalize(token)):
            chars.append((token_index, char_index, char))
    left = 0
    right = len(chars) - 1
    trace: list[dict] = []
    while left < right:
        lt, lc, lv = chars[left]
        rt, rc, rv = chars[right]
        state = {"left": [lt, lc], "right": [rt, rc], "left_char": lv,
                 "right_char": rv, "matched": lv == rv,
                 "internal_word_center_possible": lt == rt}
        trace.append(state)
        if lv != rv:
            return {"closed": False, "states": len(trace),
                    "first_debt": {"left": lv, "right": rv,
                                   "left_position": left, "right_position": right},
                    "trace": trace}
        left += 1
        right -= 1
    return {"closed": bool(chars), "states": len(trace), "first_debt": None,
            "trace": trace}


def audit(text: str) -> dict:
    letters = normalize(text)
    mismatches = [(i, len(letters) - 1 - i, letters[i], letters[-1 - i])
                  for i in range(len(letters) // 2)
                  if letters[i] != letters[-1 - i]]
    return {"letters": len(letters), "exact": bool(letters) and not mismatches,
            "mismatches": mismatches[:8],
            "sha256": hashlib.sha256(letters.encode()).hexdigest()}


def tiny_fixtures() -> list[tuple[str, ...]]:
    # Equal and unequal clause partitions, centers inside a lexical token, and
    # one-character negatives.  These are synthetic controls, never candidates.
    return [
        ("ab", "cba"),       # abcba; unequal two-token partition
        ("a", "bc", "ba"),  # abcba; center inside ``bc``
        ("ab", "c", "ba"),  # abcba; center is a singleton token
        ("a", "bcb", "a"),  # abcba; center inside the middle token
        ("ab", "cbb"),       # negative
        ("a", "bd", "a"),   # negative
    ]


def existing_prose_derivations() -> list[tuple[str, ...]]:
    # Reuse the existing recursive coordination grammar; do not add a bank or
    # import the 38-letter seed.  Distinct content roles keep each realization
    # ordinary and non-repetitive.
    clauses = [
        ("the", "quiet", "pilot", "maps", "a", "coast"),
        ("a", "patient", "gardener", "tends", "the", "orchard"),
        ("the", "young", "scholar", "reads", "a", "ledger"),
        ("a", "careful", "keeper", "packs", "the", "crates"),
        ("the", "old", "sailor", "marks", "a", "harbor"),
    ]
    out: list[tuple[str, ...]] = []
    for count in (1, 2, 3):
        for chosen in itertools.permutations(clauses, count):
            if len({item[2] for item in chosen}) != count:
                continue
            tokens: list[str] = []
            for index, clause in enumerate(chosen):
                if index:
                    tokens.append("and")
                tokens.extend(clause)
            rendered = " ".join(tokens)
            if 39 <= len(normalize(rendered)) <= 60:
                out.append(tuple(tokens))
    return out


def main() -> dict:
    differential = []
    for tokens in tiny_fixtures():
        text = " ".join(tokens)
        product = crossing_product(tokens)
        differential.append({"tokens": tokens, "text": text,
                             "oracle_exact": oracle(text),
                             "product_exact": product["closed"],
                             "product": product})
    if any(row["oracle_exact"] != row["product_exact"] for row in differential):
        raise AssertionError("midpoint product disagrees with full-tape oracle")

    candidates = []
    best = []
    for tokens in existing_prose_derivations():
        text = " ".join(tokens)
        product = crossing_product(tokens)
        row = {"text": text, "tokens": tokens, "letters": len(normalize(text)),
               "exact": product["closed"], "product": product, "audit": audit(text),
               "provenance": {"source": "existing recursive coordination grammar",
                              "new_bank": False, "seed_imported": False,
                              "word_order_mirror": False, "finished_tape_reversal": False}}
        if row["exact"]:
            candidates.append(row)
        elif len(best) < 12:
            best.append(row)

    report = {
        "experiment_id": ID,
        "signature": SIG,
        "status": "completed_exact" if candidates else "completed_no_exact_closure",
        "method": "outside-in cursor product over one shared derivation; lexical and clause boundaries are ordinary transitions, and closure is allowed at any midpoint",
        "differential": {"fixtures": differential, "passed": True,
                         "oracle": "normalize(text) == normalize(text)[::-1]",
                         "coverage": ["equal partition", "unequal partition", "internal-word center", "one-character negative"]},
        "grammar": {"source": "experiments/recursive_coordination_char_product_20260917.py::CLAUSES",
                     "target_letters": [39, 60], "derivations": len(existing_prose_derivations())},
        "candidates": candidates,
        "diagnostic_witnesses": best,
        "stats": {"differential_cases": len(differential),
                  "prose_derivations": len(existing_prose_derivations()),
                  "exact": len(candidates),
                  "longest_letters": max((row["letters"] for row in best + candidates), default=0)},
        "reader_gate": {"status": "human_blind_review_required" if candidates else "not_triggered",
                        "controls": "intact and shuffled prose are required before any readability claim"},
        "next_repair": "keep midpoint crossing and add one held-out semantic frame at the first residual; do not force equal clause halves or enlarge the bank alone",
        "provenance": {"generator": str(Path(__file__).relative_to(ROOT)),
                       "audit": "independent cursor trace plus full-tape two-pointer audit",
                       "shortcuts_rejected": ["catalogue text", "seed wrapper", "word-order symmetry", "repeated unit"]},
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["stats"]))
    return report


if __name__ == "__main__":
    main()
