"""Centered ABBA paragraph topology with a single semantic pivot.

The topology is A1, B1, C, B2, A2: outer clauses return to the same
discourse roles, while the center is a separately authored pivot.  ABBA is
only semantic structure; acceptance still requires one global character
palindrome.  No clause is copied or reversed to manufacture a candidate.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "centered-abba-pivot-20260921.json"
ID = "centered-abba-pivot-20260921"


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def audit(text: str) -> dict:
    """Two independent normalizers plus a two-pointer check."""
    tape = letters(text)
    second = "".join(c.lower() for c in text if c.isascii() and c.isalpha())
    i, j = 0, len(second) - 1
    while i < j and second[i] == second[j]:
        i += 1
        j -= 1
    return {
        "letters": len(tape), "normalized": tape,
        "exact": bool(tape) and tape == tape[::-1],
        "independent_pointer_exact": bool(second) and i >= j,
        "normalizers_agree": tape == second,
        "sha256": hashlib.sha256(tape.encode()).hexdigest(),
        "reverse_sha256": hashlib.sha256(tape[::-1].encode()).hexdigest(),
    }


def novelty(text: str, known: list[str]) -> dict:
    tape = letters(text)
    known_tapes = {letters(x) for x in known}
    repeated = len(tape) != len(set(tape))  # diagnostic only, never a gate
    return {"matches_known_tape": tape in known_tapes,
            "repeated_letters_diagnostic": repeated,
            "known_count": len(known_tapes)}


def render(parts: tuple[str, ...]) -> str:
    # Semicolons expose the five independently authored clauses to a reader;
    # they are punctuation only and cannot contribute letters to the audit.
    return "; ".join(parts[:-1]) + "; " + parts[-1]


def run() -> dict:
    # Each alternative is independently authored prose, not a transformation
    # of another slot.  A/B roles recur semantically, but wording differs.
    a1 = [
        "The gardener marked the north gate",
        "The cartographer traced the river bend",
    ]
    b1 = [
        "the patient keeper asked for a lantern",
        "the harbor pilot asked for a chart",
    ]
    center = [
        "At noon the quiet bell answered from the tower",
        "At dusk the old beacon answered across the water",
    ]
    b2 = [
        "the pilot carried the chart toward the harbor",
        "the keeper carried the lantern toward the gate",
    ]
    a2 = [
        "and the gardener watched the northern path.",
        "and the cartographer watched the turning river.",
    ]
    records = []
    for parts in itertools.product(a1, b1, center, b2, a2):
        text = render(parts)
        check = audit(text)
        records.append({"text": text, "length": check["letters"],
                        "topology": ["A1", "B1", "C", "B2", "A2"],
                        "provenance": {
                            "generator": ID, "slot_sources": "authored finite grammar",
                            "center_independently_authored": True,
                            "reused_clause": False,
                            "reverse_or_catalogue_source": False,
                        }, "audit": check,
                        "novelty": novelty(text, [])})
    exact = [r for r in records if r["audit"]["exact"] and
             r["audit"]["independent_pointer_exact"] and
             r["audit"]["normalizers_agree"] and
             not r["novelty"]["matches_known_tape"]]
    return {"id": ID, "method": "semantic ABBA around an independently authored center pivot",
            "topology": "A1 B1 C B2 A2", "candidate_count": len(records),
            "exact_admitted_count": len(exact), "max_length": max(r["length"] for r in records),
            "records": records, "exact_admitted": exact,
            "next_repair": "condition B2 lexical choices on the residual after A1+B1+C; do not duplicate this Cartesian sweep"}


if __name__ == "__main__":
    result = run()
    RUN.parent.mkdir(exist_ok=True)
    RUN.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("candidate_count", "exact_admitted_count", "max_length", "next_repair")}, indent=2))
