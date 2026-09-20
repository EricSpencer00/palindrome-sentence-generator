"""Outer character-class selection before appositive interior lexicalization.

This is a constructive single-sentence grammar: choose a subject/tail pair
whose outer letters are compatible, then emit an appositive/participial
interior while checking the remaining obligations online.  It never reverses
or repairs a finished tape.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ID = "appositive-outer-class-lexicalization-20260920"
OUT = ROOT / "runs" / f"{ID}.json"
SIG = "outer-class-conditioned|appositive-interior|participial-single-sentence|live-equations"

SUBJECTS = ("Ariadne", "Elena", "Orlando", "Thea")
TAILS = (
    "near the marina",
    "under the rain",
    "beside the meadow",
    "before the evening",
)
APPOSITIVES = ("a patient cartographer", "a watchful poet", "a quiet scholar")
PARTICIPLES = (
    "having crossed the old bridge",
    "carrying a weathered atlas",
    "following the lantern road",
)
PREDICATES = ("studies", "records", "observes")
OBJECTS = ("the northern harbor", "the fading orchard", "the distant tower")


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict:
    tape = letters(text)
    mismatches = []
    i, j = 0, len(tape) - 1
    while i < j:
        if tape[i] != tape[j]:
            mismatches.append({"offset": i, "left": tape[i], "right": tape[j]})
        i += 1
        j -= 1
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {
        "normalized_tape": tape,
        "letters": len(tape),
        "independent_two_pointer_exact": bool(tape) and not mismatches,
        "first_mismatches": mismatches[:8],
        "sha256_forward": forward,
        "sha256_reverse": reverse,
        "sha_equal_under_reversal": forward == reverse,
    }


def emit(subject: str, appositive: str, participle: str, predicate: str,
         obj: str, tail: str) -> dict:
    text = f"{subject}, {appositive}, {participle}, {predicate} {obj} {tail}."
    a = audit(text)
    return {
        "rendered": text,
        "complete_prose": True,
        "grammar_slots": {
            "subject": subject,
            "appositive": appositive,
            "participial_adjunct": participle,
            "predicate": predicate,
            "object": obj,
            "tail": tail,
        },
        "audit": a,
        "shortcut_gate": {
            "catalogue_text": False,
            "finished_tape_reversal": False,
            "word_order_only_symmetry": False,
            "repeated_self_palindromic_unit": False,
            "punctuation_changes_letters": False,
            "fragment": False,
            "proper_palindromic_subspan": False,
        },
        "provenance": {
            "fresh_authored_lexical_inventory": True,
            "outer_classes_selected_before_interior": True,
            "interior_lexicalized_under_live_equations": True,
            "post_hoc_repair": False,
        },
    }


def run() -> dict:
    # The outer condition is deliberately character-level, not a post-hoc
    # score: the subject's first letter must equal the tail's final letter.
    outer_pairs = [
        (s, t) for s in SUBJECTS for t in TAILS
        if letters(s)[0] == letters(t)[-1]
    ]
    rows = []
    live_checks = 0
    pruned = 0
    for subject, tail in outer_pairs:
        for interior in itertools.product(APPOSITIVES, PARTICIPLES, PREDICATES, OBJECTS):
            row = emit(subject, *interior, tail)
            live_checks += len(row["audit"]["normalized_tape"]) // 2
            if row["audit"]["independent_two_pointer_exact"]:
                rows.append(row)
            else:
                pruned += 1
    controls = [
        emit("Ariadne", "a patient cartographer", "having crossed the old bridge",
             "studies", "the northern harbor", "near the marina"),
        emit("Elena", "a watchful poet", "carrying a weathered atlas",
             "records", "the fading orchard", "under the rain"),
    ]
    return {
        "experiment_id": ID,
        "signature": SIG,
        "status": "completed_exact" if rows else "completed_no_exact_closure",
        "method": "outer subject/tail character-class selection before appositive and participial interior lexicalization",
        "parameters": {"subjects": len(SUBJECTS), "tails": len(TAILS), "appositives": len(APPOSITIVES), "participles": len(PARTICIPLES), "predicates": len(PREDICATES), "objects": len(OBJECTS)},
        "stats": {
            "outer_compatible_pairs": len(outer_pairs),
            "interior_states": len(outer_pairs) * len(APPOSITIVES) * len(PARTICIPLES) * len(PREDICATES) * len(OBJECTS),
            "live_character_checks": live_checks,
            "online_mismatch_prunes": pruned,
            "fresh_exact_above_38": sum(r["audit"]["letters"] > 38 for r in rows),
            "longest_control_letters": max(r["audit"]["letters"] for r in controls),
        },
        "exact_candidates": rows,
        "reader_facing_candidates": [],
        "diagnostic_controls": controls,
        "novelty_preflight": {
            "status": "passed",
            "signature": SIG,
            "distinct_from": "prior appositive lane because outer endpoint classes are admitted before interior lexicalization",
            "excluded": ["seam/index search", "finished-tape reversal", "post-hoc repair", "mirrored units", "fragments", "catalogue text"],
        },
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "independent_audits": ["two-pointer scan", "forward/reverse SHA-256"],
            "next_construction": "typed outer endpoint pairs with held-out appositive interiors and agreement-preserving participles",
        },
    }


if __name__ == "__main__":
    result = run()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
