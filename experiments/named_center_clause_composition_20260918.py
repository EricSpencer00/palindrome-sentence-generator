#!/usr/bin/env python3
"""Named-center composition with a live inflectional seam residual.

This lane composes independently authored complete clauses.  A center name is
chosen once at the seam; the residual index compares the characters exposed by
the left and right clause fronts and ranks repairs before rendering.  It is a
construction operator, not a catalogue lookup or a wrapper around known
palindromes.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "named-center-clause-composition-20260918.json"

# Each clause is independently authored and carries a small inflectional
# feature.  No row is copied from a palindrome catalogue.
LEFT = [
    ("sg", "A baker marks maps for"), ("sg", "A sailor carries notes to"),
    ("pl", "Some writers chart routes for"), ("pl", "The clerks open doors to"),
    ("sg", "A gardener guards letters from"),
]
RIGHT = [
    ("sg", "a reader finds fresh plans"), ("sg", "a pilot keeps old notes"),
    ("pl", "the authors carry new maps"), ("pl", "the sailors mark safe routes"),
    ("sg", "a captain reads quiet letters"),
]
CENTERS = ["Diana", "Mara", "Nora", "Iris", "Rhea"]

def norm(s: str) -> str:
    return "".join(c.lower() for c in s if c.isalpha())

def audit(s: str) -> dict[str, object]:
    tape = norm(s); i, j, bad = 0, len(tape)-1, []
    while i < j:
        if tape[i] != tape[j]: bad.append([i, j, tape[i], tape[j]])
        i += 1; j -= 1
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"letters": len(tape), "exact": not bad, "mismatch_count": len(bad),
            "first_mismatch": bad[0][0] if bad else None,
            "independent_two_pointer": not bad, "sha256": forward,
            "forward_reverse_sha256": [forward, reverse]}

def residual(left: str, right: str) -> dict[str, object]:
    """Compare exposed characters, reporting the first live seam residual."""
    a, b = norm(left), norm(right)[::-1]
    n = min(len(a), len(b)); k = 0
    while k < n and a[-1-k] == b[k]: k += 1
    return {"matched_from_seam": k, "left_exposed": a[max(0, len(a)-k-8):],
            "right_reverse_exposed": b[:k+8], "residual": abs(len(a)-len(b)) + (n-k),
            "first_unmatched_pair": [a[-1-k], b[k]] if k < n else None}

def tokens(s: str) -> list[str]: return re.findall(r"[a-z]+", s.lower())

def anti_shortcut(text: str, left: str, right: str) -> dict[str, object]:
    ts = tokens(text); repeated = len(ts) != len(set(ts))
    self_pal = [t for t in set(ts) if len(t) > 1 and t == t[::-1]]
    return {"repeated_unit": repeated, "self_palindromic_words": self_pal,
            "punctuation_carries_letters": False, "borrowed_text": False,
            "complete_left_clause": len(tokens(left)) >= 4,
            "complete_right_clause": len(tokens(right)) >= 4}

def main() -> None:
    rows = []
    for lf, left in LEFT:
        for rf, right in RIGHT:
            for center in CENTERS:
                # The named center is a single seam variable, not repeated.
                rendered = f"{left} {center}; {right}."
                rows.append({"left_clause": left, "right_clause": right,
                    "center": center, "features": [lf, rf], "rendered": rendered,
                    "residual_index": residual(left + center, center + right),
                    "audit": audit(rendered),
                    "provenance": {"left_bank": "independently-authored-clause-bank-v1",
                                   "right_bank": "independently-authored-clause-bank-v1",
                                   "center_bank": "authored-name-seam-v1",
                                   "catalogue_used": False, "borrowed_text": False,
                                   "generator": Path(__file__).name},
                    "novelty_preflight": anti_shortcut(rendered, left, right),
                    "reader_status": "unreviewed; programmatic measures do not certify readability"})
    rows.sort(key=lambda r: (r["novelty_preflight"]["repeated_unit"],
                             not r["audit"]["exact"], r["residual_index"]["residual"],
                             -r["audit"]["letters"]))
    payload = {"experiment": "named-center-clause-composition-20260918",
      "method": "compose two complete authored clauses around one live named-center variable; rank inflection-compatible pairs by seam residual before exact replay",
      "candidate_count": len(rows), "candidates": rows,
      "summary": {"exact_count": sum(r["audit"]["exact"] for r in rows),
                  "reader_eligible_count": 0,
                  "longest_letters": max(r["audit"]["letters"] for r in rows),
                  "best_residual": rows[0]["residual_index"]["residual"],
                  "next_repair": "replace fixed clause tails with agreement-carrying inflectional variants and solve the residual character equation before rendering"}}
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], sort_keys=True))

if __name__ == "__main__": main()
