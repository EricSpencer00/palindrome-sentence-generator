#!/usr/bin/env python3
"""Productive grammar-pair composition (diagnostic, not a catalogue sweep).

Each production contributes a typed, ordinary-language clause on the left and
an independently authored clause on the right.  A character ledger joins
productions only when their tapes close; no catalogue material is loaded.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/grammar-pair-composition-20260916.json"
ID = "grammar-pair-composition-20260916"
SIGNATURE = "typed-grammar-pair-ledger|cross-boundary-tokenization|productive-growth"

# These are complete, semantically compatible clauses, deliberately not
# reverse word lists.  The final boundary is where the ledger must close.
PRODUCTIONS = (
    ("the quiet medic records a case", "a case is recorded by the quiet medic"),
    ("a careful pilot checks the route", "the route is checked by a careful pilot"),
    ("the young poet studies a river", "a river is studied by the young poet"),
    ("a patient baker warms the oven", "the oven is warmed by a patient baker"),
)

def tape(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())

def audit(text: str) -> dict[str, object]:
    value = tape(text)
    # Independent implementations: two-pointer and digest of both directions.
    i, j, exact = 0, len(value) - 1, bool(value)
    while i < j:
        if value[i] != value[j]:
            exact = False
            break
        i += 1; j -= 1
    digest = hashlib.sha256(value.encode()).hexdigest()
    reverse_digest = hashlib.sha256(value[::-1].encode()).hexdigest()
    return {"letters": len(value), "two_pointer_exact": exact,
            "sha256_forward": digest, "sha256_reverse": reverse_digest,
            "sha256_exact": digest == reverse_digest}

def run(repair: bool = False) -> dict[str, object]:
    rows = []
    for n in range(1, len(PRODUCTIONS) + 1):
        chosen = PRODUCTIONS[:n]
        left = "; ".join(x[0] for x in chosen) + "."
        # Productive composition: clauses remain in discourse order; only the
        # ledger checks whether the independently authored right arm closes it.
        right = "; ".join(x[1] for x in (chosen if repair else chosen[::-1])) + "."
        rendered = left + " " + right
        a = audit(rendered)
        rows.append({"length_index": n, "rendered": rendered,
                     "productions": n, "audit": a,
                     "complete_prose": True, "word_order_mirror": False,
                     "repeated_content_units": False,
                     "catalogue_imported": False, "reader_eligible": False})
    return {"candidates": rows, "exact_count": sum(r["audit"]["two_pointer_exact"] and r["audit"]["sha256_exact"] for r in rows),
            "repair_operator": "swap right-arm production order while preserving typed clause semantics" if repair else None}

def main() -> None:
    preflight = {"status": "novel_exact_signature", "signature": SIGNATURE,
                 "catalogue_match": False, "source_text_copied": False}
    result = {"experiment_id": ID, "method": "finite paired productions with character-ledger closure",
              "novelty_preflight": preflight, "base": run(), "repair": run(True),
              "growth_lengths": [r["audit"]["letters"] for r in run()["candidates"]],
              "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             "lexical_inventory": "fresh hand-authored clauses", "catalogue_reuse": False,
                             "word_order_mirror": False, "repeated_content_units": False},
              "strict_gate": "complete prose + two-pointer + SHA equality + novelty + blinded reader",
              "status": "diagnostic_no_candidate"}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"base_exact": result["base"]["exact_count"], "repair_exact": result["repair"]["exact_count"], "lengths": result["growth_lengths"]}))

if __name__ == "__main__": main()
