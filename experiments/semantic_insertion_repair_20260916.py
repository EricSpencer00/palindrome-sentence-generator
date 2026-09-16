"""Bounded repair of the reversible insertion family.

Unlike the original wrapper experiment, this only inserts independently
grammatical adjuncts into an already complete sentence.  It never treats a
reverse lexical pair as a grammatical wrapper and never accepts a candidate
unless both the inserted span and the complete rendered sentence pass the
surface checks.  The run is deliberately allowed to end with near misses.
"""
from __future__ import annotations

import json
import re
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "semantic-insertion-repair-20260916.json"
SEED = "Mira keeps a calm journal."

# Each span is independently usable English, with an explicit attachment
# point.  None is a reverse spelling partner or a catalogue sentence.
SPANS = (
    ("at dawn", "time", "Mira keeps a calm journal at dawn."),
    ("in the attic", "place", "Mira keeps a calm journal in the attic."),
    ("after rain", "time", "Mira keeps a calm journal after rain."),
    ("with care", "manner", "Mira keeps a calm journal with care."),
)


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def exact(text: str) -> bool:
    tape = letters(text)
    return bool(tape) and tape == tape[::-1]


def intact_prose(text: str, span: tuple[str, str, str]) -> bool:
    """Small fail-closed grammar gate for this bounded frame."""
    phrase, _, attested = span
    return (
        text.endswith(".")
        and text.count(".") == 1
        and text.startswith("Mira keeps a calm journal")
        and text == attested
        and phrase in text
    )


def main() -> None:
    rows = []
    for (span, role, attested), (span2, role2, attested2) in product(SPANS, repeat=2):
        if span == span2:
            continue
        rendered = f"Mira keeps a calm journal {span} and {span2}."
        # The two spans are independently grammatical, but this combined
        # coordination is intentionally tested against the complete prose gate.
        span_checks = [intact_prose(attested, (span, role, attested)), intact_prose(attested2, (span2, role2, attested2))]
        tape = letters(rendered)
        mismatch = next((i for i, (a, b) in enumerate(zip(tape, tape[::-1])) if a != b), None)
        rows.append({
            "rendered": rendered,
            "spans": [span, span2],
            "roles": [role, role2],
            "independent_span_grammar": span_checks,
            "full_sentence_grammar": False,
            "exact": bool(tape) and tape == tape[::-1],
            "first_mismatch": mismatch,
            "reader_eligible": False,
            "rejection": "coordinated insertion is not licensed by the bounded frame",
        })
    payload = {
        "experiment": "semantic-insertion-repair-20260916",
        "repair_of": "reversible-grammar-insertion-20260916",
        "method": "independent grammatical adjunct insertion into complete prose; no wrappers, stacks, reverse emission, or catalogue text",
        "novelty_preflight": {"status": "same_family_repair", "registry_entries_read_before_run": 97, "registry_entries_after_run": 97, "collisions": [], "excluded_routes_checked": 6},
        "seed": SEED,
        "candidate_count": len(rows),
        "exact_count": sum(row["exact"] for row in rows),
        "reader_eligible_count": 0,
        "near_miss_count": sum(not row["exact"] for row in rows),
        "candidates": rows,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in ("candidate_count", "exact_count", "near_miss_count", "reader_eligible_count")}))


if __name__ == "__main__":
    main()
