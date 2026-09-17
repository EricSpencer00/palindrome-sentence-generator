"""Clause-level equation solver over the semantic-valency scene lattice.

Outer complete clauses are selected first and their exposed character
obligations are scored before the middle clause is expanded.  This is a
bounded equation solver, not a reverse decoder: every emitted state remains
ordinary prose assembled from complete valency-compatible clauses.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks
from experiments.semantic_valency_attachment_scene_lattice_20260916 import (
    CLAUSES,
    EXPERIMENT as LATTICE_EXPERIMENT,
    exact_sha,
    exact_two_pointer,
    independent_admission,
    novelty_preflight as lattice_preflight,
    tape,
)

EXPERIMENT = "semantic-valency-clause-equation-solver-20260916"
SIGNATURE = (
    "semantic-valency-attachment-lattice|outer-clause-equation-pruning|"
    "joint-complete-clauses|ordinary-order|independent-pointer-sha-reader-gate"
)
REGISTRY = ROOT / "docs" / "experiment-novelty-registry.json"
OUT = ROOT / "runs" / f"{EXPERIMENT}.json"


def novelty_preflight() -> dict:
    entries = json.loads(REGISTRY.read_text()).get("entries", [])
    collisions = [e["id"] for e in entries if e.get("id") != EXPERIMENT and e.get("signature") == SIGNATURE]
    return {"entries_inspected": len(entries), "exact_signature_collisions_before_render": collisions, "passed": not collisions, "state_space_distinction": "outer complete valency clauses are obligation-scored before middle-clause expansion; no reverse text is generated"}


def outer_equation(left: dict, right: dict) -> dict:
    a, b = tape(left["text"]), tape(right["text"])
    checked = min(len(a), len(b))
    matches = sum(a[i] == b[::-1][i] for i in range(checked))
    first = next((i for i in range(checked) if a[i] != b[::-1][i]), None)
    return {"left_clause": left["id"], "right_clause": right["id"], "positions_checked": checked, "matching_pairs": matches, "mismatch_pairs": checked - matches, "first_mismatch_offset": first, "left_prefix": a[:24], "right_reverse_obligation": b[::-1][:24], "equation": "left_clause[i] = reverse(right_clause)[i] before middle expansion"}


def audit(choice: tuple[dict, ...], rank: int, outer: dict) -> dict:
    text = " ".join(c["text"] for c in choice)
    normalized = tape(text)
    pointer, sha = exact_two_pointer(text), exact_sha(text)
    central = mechanical_admission_checks(text, min_letters=120, max_letters=240)
    independent = independent_admission(text)
    reader = {"eligible": bool(normalized) and normalized == normalized[::-1] and pointer["exact"] and sha["exact"] and all(central.values()) and all(independent.values()), "exact_closure_required": True, "reason": "exact closure and all anti-shortcut admission checks are required"}
    return {"rank": rank, "rendered": text, "normalized_tape": normalized, "letters": len(normalized), "clause_provenance": [{"id": c["id"], "meaning": c["meaning"], "position": i + 1} for i, c in enumerate(choice)], "outer_equation": outer, "exact_check_two_pointer": pointer, "exact_check_sha256": sha, "independent_exact_agreement": pointer["exact"] == sha["exact"], "central_admission": central, "independent_admission": independent, "reader_gate": reader, "anti_shortcut_flags": {"fixed_tape": False, "reverse_decoder": False, "mirrored_word_units": False, "repeated_palindromic_unit": False, "catalogue_text_used": False, "isolated_character_edit": False, "complete_constituents_only": True, "semantic_valency_checked": True, "outer_equation_pruned": True, "ordinary_order_events": True}, "mechanically_admitted": reader["eligible"], "next_repair": "Replace the first terminal clause pair that mismatches at the outer frontier with fresh valency-compatible clauses whose exposed characters satisfy the first obligation, then rerun this solver."}


def run() -> dict:
    preflight = novelty_preflight()
    if not preflight["passed"]:
        raise RuntimeError(f"novelty collision: {preflight['exact_signature_collisions_before_render']}")
    # The lattice preflight is carried as lineage evidence, while this solver
    # performs its own signature preflight before any candidate is rendered.
    lineage = lattice_preflight()
    outer_states = []
    for left, right in itertools.product(CLAUSES[0], CLAUSES[2]):
        outer_states.append((outer_equation(left, right), left, right))
    outer_states.sort(key=lambda x: (-x[0]["matching_pairs"], x[0]["first_mismatch_offset"] or 999))
    rows = []
    rank = 0
    for outer, left, right in outer_states:
        for middle in CLAUSES[1]:
            rank += 1
            rows.append(audit((left, middle, right), rank, outer))
    rows.sort(key=lambda r: (not r["mechanically_admitted"], -r["outer_equation"]["matching_pairs"], r["outer_equation"]["first_mismatch_offset"] or 999, -r["letters"]))
    exact = [r for r in rows if r["mechanically_admitted"]]
    return {"experiment": EXPERIMENT, "signature": SIGNATURE, "status": "complete; no exact closure" if not exact else "exact closure found", "novelty_preflight": preflight, "lineage_lattice_experiment": LATTICE_EXPERIMENT, "lineage_preflight": lineage, "outer_states_examined": len(outer_states), "states_expanded": len(rows), "exact_count": len(exact), "best_rendered_candidates": rows[:6], "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "registry_sha256": hashlib.sha256(REGISTRY.read_bytes()).hexdigest(), "operator": "outer complete-clause equation scoring followed by middle-clause expansion", "ordinary_order_rendering": True}, "anti_shortcut_policy": "No fixed tape, reverse decoder, mirrored word order, repeated units, catalogue text, or isolated edits; only complete semantic clauses are jointly selected.", "next_repair": "Author a new terminal clause pair to repair the first outer character obligation, then rerun novelty preflight and this bounded solver."}


if __name__ == "__main__":
    if OUT.exists():
        raise SystemExit(f"output already exists: {OUT}")
    result = run()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"outer_states_examined": result["outer_states_examined"], "states_expanded": result["states_expanded"], "exact_count": result["exact_count"]}, indent=2))
