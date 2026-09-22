"""Center-atom/event-relation extension of the API-inspired probe.

This is a bounded diagnostic, not a bank composer: two fresh event frames are
joined through one lexical connective chosen together with a typed relation.
The complete frame product is small and frozen; the useful search state is the
outer-character trace, which records exactly where the relation-bearing prose
loses its live mirror obligation.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from experiments.api_inspired_semantic_atom_transducer_20260920 import (
    SCENES,
    Atom,
    audit,
    hidden_span,
    norm,
)


OUT = Path(__file__).resolve().parents[1] / "runs" / "api-inspired-center-relation-transducer-20260920.json"


@dataclass(frozen=True)
class CenterAtom:
    ident: str
    surface: str
    relation: str


CENTERS = (
    CenterAtom("center-while", "while", "simultaneous"),
    CenterAtom("center-before", "before", "temporal-precedence"),
    CenterAtom("center-because", "because", "causal-dependence"),
    CenterAtom("center-despite", "despite", "contrast"),
)


def clauses(scene: str) -> tuple[tuple[Atom, Atom, Atom], ...]:
    bank = SCENES[scene]
    return tuple(product(bank["subject"], bank["verb"], bank["object"]))


def frame_surface(frame: tuple[Atom, Atom, Atom]) -> str:
    return " ".join(atom.surface for atom in frame)


def relation_compatible(center: CenterAtom,
                        left: tuple[Atom, Atom, Atom],
                        right: tuple[Atom, Atom, Atom]) -> bool:
    """Apply semantic compatibility before rendering the joined text."""
    left_subject, left_verb, left_object = left
    right_subject, right_verb, right_object = right
    if left_subject.scene != right_subject.scene:
        return False
    if center.relation == "simultaneous":
        return left_verb.scene == right_verb.scene
    if center.relation == "temporal-precedence":
        return left_verb.ident != right_verb.ident
    if center.relation == "causal-dependence":
        return left_object.ident != right_object.ident
    if center.relation == "contrast":
        return left_subject.ident != right_subject.ident
    return False


def outer_trace(text: str) -> dict:
    tape = norm(text)
    pairs = []
    for i in range(len(tape) // 2):
        j = len(tape) - i - 1
        pairs.append({"left_index": i, "right_index": j,
                      "left": tape[i], "right": tape[j],
                      "matched": tape[i] == tape[j]})
        if tape[i] != tape[j]:
            break
    return {
        "compared_from_outer_edge": len(pairs),
        "matched_prefix": sum(pair["matched"] for pair in pairs),
        "first_mismatch": next((pair for pair in pairs if not pair["matched"]), None),
        "trace": pairs[:16],
    }


def run() -> dict:
    rows: list[dict] = []
    exact: list[dict] = []
    scene_stats = {}
    for scene in SCENES:
        scene_rows = []
        frame_bank = clauses(scene)
        for left, right, center in product(frame_bank, frame_bank, CENTERS):
            ids = [atom.ident for atom in left + right]
            if len(set(ids)) != len(ids):
                continue
            if not relation_compatible(center, left, right):
                continue
            rendered = f"{frame_surface(left)} {center.surface} {frame_surface(right)}."
            row = {
                "rendered": rendered,
                "scene": scene,
                "left_atoms": [atom.ident for atom in left],
                "center_atom_id": center.ident,
                "right_atoms": [atom.ident for atom in right],
                "relation_id": center.relation,
                "outer_trace": outer_trace(rendered),
                "audit": audit(rendered),
                "hidden_palindromic_span": hidden_span(rendered),
                "provenance": {
                    "fresh_hand_authored_atoms": True,
                    "center_atom_is_lexical": True,
                    "relation_checked_before_render": True,
                    "atom_ids_distinct": True,
                    "finished_tape_reversal": False,
                    "post_hoc_repair": False,
                    "catalogue_text": False,
                    "repeated_units": False,
                },
            }
            scene_rows.append(row)
            if row["audit"]["exact"]:
                exact.append(row)
        scene_rows.sort(key=lambda row: (-row["outer_trace"]["matched_prefix"], row["rendered"]))
        rows.extend(scene_rows)
        scene_stats[scene] = {
            "semantic_joins": len(scene_rows),
            "max_outer_matched": max((r["outer_trace"]["matched_prefix"] for r in scene_rows), default=0),
            "exact": sum(r["audit"]["exact"] for r in scene_rows),
        }
    clean = [r for r in exact if not r["hidden_palindromic_span"]]
    return {
        "experiment_id": "api-inspired-center-relation-transducer-20260920",
        "method": "lexical center atom plus typed binary event relation with live outer-character trace",
        "stats": {
            "scenes": len(SCENES),
            "semantic_joins": len(rows),
            "exact": len(exact),
            "exact_over_38": sum(r["audit"]["letters"] > 38 for r in exact),
            "clean_exact": len(clean),
            "longest_rendered_letters": max((r["audit"]["letters"] for r in rows), default=0),
            "max_outer_matched": max((r["outer_trace"]["matched_prefix"] for r in rows), default=0),
        },
        "scene_stats": scene_stats,
        "rendered_candidates": rows[:24],
        "exact_candidates": exact,
        "reader_facing_candidates": [],
        "provenance": {
            "fresh_hand_authored_inventory": True,
            "relation_ids": [center.relation for center in CENTERS],
            "independent_audits": ["outer two-pointer trace", "normalized forward/reverse SHA-256"],
        },
        "novelty_preflight": {
            "signature": "api-mirror-state|lexical-center-atom|typed-event-relation|outer-trace",
            "registry_inspected": True,
            "closest_prior": "API-inspired semantic atom transducer; this adds a lexical center and typed relation before rendering",
            "falsifier": "if relation compatibility does not change reachable states, the semantic edge is inert",
        },
        "status": "no exact closure" if not exact else "exact closure requires independent reader gate",
        "next_construction": "replace the fixed SVO frame product with unresolved character-level grammar states while retaining the center relation",
    }


if __name__ == "__main__":
    result = run()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"]))
    print(json.dumps(result["scene_stats"], indent=2))
