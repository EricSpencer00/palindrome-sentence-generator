"""Bounded relative-clause extension of the typed zipper.

This is a distinct repair of the 38-letter anchor lane: a final authored
relative clause is kept as a live grammatical constituent on each side while
the zipper matches its character debt.  It is not a catalogue wrapper or a
post-hoc reversal.  The run is deliberately small so a zero is useful
evidence for the next repair rather than another broad lexical sweep.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import typed_clause_zipper_20260919 as base

EXPERIMENT_ID = "zipper-relative-clause-extension-20260919"

# These are intact, independently authored English relative clauses.  Their
# subject is implicit and they are only admitted in the P constituent here;
# no sentence or palindrome catalogue is imported.
RELATIVE_CLAUSES = tuple(
    base.opt(text) for text in (
        "who reads the notes", "who keeps the book", "who marks the page",
        "who sings at dawn", "who guards the gate", "who writes a poem",
        "who hears the bell", "who follows the sailor",
    )
)


def run() -> dict[str, object]:
    old = base.ADJUNCTS
    base.ADJUNCTS = old + RELATIVE_CLAUSES
    try:
        rows, nodes = base.search(("S", "V", "O", "P"), ("S", "V", "O", "P"),
                                  max_nodes=250_000)
    finally:
        base.ADJUNCTS = old
    unique = {row["audit"]["normalized"]: row for row in rows}
    exact = sorted(unique.values(), key=lambda row: -row["audit"]["letters"])
    return {
        "experiment_id": EXPERIMENT_ID,
        "method": "two-sided typed zipper with live authored relative-clause slot",
        "relative_clause_bank": [x.text for x in RELATIVE_CLAUSES],
        "stats": {"nodes": nodes, "exact": len(exact),
                  "longest_exact_letters": max((x["audit"]["letters"] for x in exact), default=0)},
        "candidates": exact,
        "independent_audit": ["base two-pointer normalized tape", "base forward/reverse SHA-256"],
        "provenance": {"catalogue_imported": False, "finished_tape_reversed": False,
                       "word_order_mirror": False, "reader_status": "not_run"},
        "next_repair": "Carry the relative clause's internal verb/object as separate typed slots; the current constituent-level debt cannot expose its residual character domains.",
        "reader_gate": "closed; no exact candidate reached it",
    }


if __name__ == "__main__":
    result = run()
    out = ROOT / "runs" / (EXPERIMENT_ID + ".json")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
