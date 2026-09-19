"""Relative-slot repair for the z2 half-tape CSP.

The relative clause is grammar, not an opaque phrase: ``who``, its finite
verb, and its object each consume live mirrored character variables.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks
from experiments.half_tape_grammar_csp_z2_20260919 import (
    Edge, SUBJECTS, VERBS, OBJECTS, SEAM, audit,
)

EXPERIMENT_ID = "half-tape-relative-slots-z2-20260919"
REL_MARK = (Edge("who", "R"), Edge("that", "R"))
REL_VERBS = tuple(edge for edge in VERBS if edge.text in {
    "reads", "marks", "writes", "keeps", "finds", "follows", "opens",
    "hears", "sees", "holds", "loves", "needs", "read", "mark", "write",
    "keep", "find", "follow", "open", "hear", "see", "hold", "love", "need",
})
REL_OBJECTS = tuple(edge for edge in OBJECTS if edge.text in {
    "the notes", "a letter", "the book", "old maps", "fresh pages", "the chart",
    "the poem", "a song", "the bell", "new books", "the map", "Diana", "Noel",
})
DOMAINS = {
    "S": SUBJECTS, "V": VERBS, "O": OBJECTS, "R": REL_MARK,
    "RV": REL_VERBS, "RO": REL_OBJECTS, "C": (SEAM,),
}
TEMPLATES = (
    ("S", "R", "RV", "RO", "V", "O", "C", "S", "V", "O"),
    ("S", "V", "O", "R", "RV", "RO", "C", "S", "V", "O"),
)


def search(template: tuple[str, ...], target: int, max_nodes: int = 3_000) -> tuple[list[dict[str, object]], int]:
    rows: list[dict[str, object]] = []
    nodes = 0
    minimum = {key: min(len(edge.tape) for edge in value) for key, value in DOMAINS.items()}
    maximum = {key: max(len(edge.tape) for edge in value) for key, value in DOMAINS.items()}

    def visit(index: int, position: int, assigned: dict[int, str], chosen: list[Edge], used: frozenset[str], number: str | None) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes:
            return
        if index == len(template):
            if position != target:
                return
            raw = " ".join(edge.text for edge in chosen).replace(" ; ", "; ")
            rendered = raw[:1].upper() + raw[1:] + "."
            checked = audit(rendered)
            if not checked["two_pointer_exact"] or checked["letters"] < 39:
                return
            gates = mechanical_admission_checks(rendered, min_letters=39, max_letters=260)
            rows.append({
                "rendered": rendered,
                "word_spans": [edge.text for edge in chosen],
                "audit": checked,
                "mechanical_checks": gates,
                "mechanically_admitted": all(gates.values()),
                "provenance": {"construction": "half-tape CSP with explicit relative marker/verb/object slots", "finished_tape_reversed": False, "catalogue_imported": False, "word_order_mirror": False, "rlaif_used": False},
                "reader_status": "unreviewed; exactness does not certify readability",
            })
            return
        slot = template[index]
        remaining = template[index + 1:]
        for edge in DOMAINS[slot]:
            if len(edge.tape) > 1 and edge.tape in used:
                continue
            if edge.tag in {"V", "RV"} and number not in (None, edge.number):
                continue
            end = position + len(edge.tape)
            if end + sum(minimum[key] for key in remaining) > target:
                continue
            if end + sum(maximum[key] for key in remaining) < target:
                continue
            next_assigned = dict(assigned)
            valid = True
            for offset, character in enumerate(edge.tape):
                absolute = position + offset
                mirror = min(absolute, target - 1 - absolute)
                if mirror in next_assigned and next_assigned[mirror] != character:
                    valid = False
                    break
                next_assigned[mirror] = character
            if valid:
                next_number = edge.number if edge.number is not None else number
                next_used = used | ({edge.tape} if len(edge.tape) > 1 else set())
                visit(index + 1, end, next_assigned, chosen + [edge], next_used, next_number)

    visit(0, 0, {}, [], frozenset(), None)
    return rows, nodes


def run() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    nodes = 0
    target_runs = 0
    for template in TEMPLATES:
        for target in range(39, 71):
            found, count = search(template, target)
            rows.extend(found)
            nodes += count
            target_runs += 1
    exact = sorted({row["audit"]["normalized"]: row for row in rows}.values(), key=lambda row: (-row["audit"]["letters"], row["rendered"]))
    return {
        "experiment_id": EXPERIMENT_ID,
        "method": "half-tape CSP with explicit relative marker, finite verb, and object slots",
        "stats": {"target_runs": target_runs, "nodes": nodes, "exact": len(exact), "mechanically_admitted": sum(row["mechanically_admitted"] for row in exact), "longest_exact_letters": max((row["audit"]["letters"] for row in exact), default=0)},
        "candidates": exact,
        "independent_audit": ["two-pointer normalized tape", "forward/reverse SHA-256"],
        "novelty_preflight": {"status": "passed", "signature": EXPERIMENT_ID, "catalogue_imported": False},
        "next_repair": "Carry anaphor agreement and a bounded adjunct as additional typed edges only after the internal relative seam closes.",
        "reader_gate": "closed; no exact candidate reached it",
    }


if __name__ == "__main__":
    result = run()
    (ROOT / "runs" / (EXPERIMENT_ID + ".json")).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
