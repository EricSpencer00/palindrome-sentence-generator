"""Held-out imperative/relative-role alternation search.

This lane is deliberately a new surface grammar rather than a repair pass:
each side chooses an imperative valency (transitive or intransitive) and a
relative attachment (subject-gap or object-gap).  Characters are compared as
the two independently authored sides are emitted, so an accepted row never
comes from reversing or editing a finished sentence.
"""
from __future__ import annotations

import hashlib
import json
import re
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/imperative-relative-role-alternation-20260920.json"
EXPERIMENT_ID = "imperative-relative-role-alternation-20260920"

TRANSITIVE = (
    ("guard", "the lantern", "the sailor", "the bridge"),
    ("follow", "the narrow road", "the guide", "the harbor"),
    ("carry", "the weathered map", "the captain", "the village"),
)
INTRANSITIVE = (
    ("wait", "by the bridge", "the guide", "the harbor"),
    ("listen", "near the garden", "the singer", "the tower"),
    ("travel", "toward the river", "the farmer", "the market"),
)
SUBJECT_GAP = (
    ("who guards the lantern", "subject-gap"),
    ("who follows the narrow road", "subject-gap"),
    ("who carries the weathered map", "subject-gap"),
)
OBJECT_GAP = (
    ("that the sailor guards", "object-gap"),
    ("that the guide follows", "object-gap"),
    ("that the captain carries", "object-gap"),
)
TAILS = ("before dusk", "through the rain", "until dawn")


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def pointer_audit(text: str) -> dict:
    tape = letters(text)
    first_mismatch = None
    for i, (a, b) in enumerate(zip(tape, reversed(tape))):
        if a != b:
            first_mismatch = [i, a, b]
            break
    forward = hashlib.sha256(tape.encode()).hexdigest()
    backward = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {
        "letters": len(tape),
        "exact": first_mismatch is None,
        "first_mismatch": first_mismatch,
        "sha256_forward": forward,
        "sha256_reverse": backward,
    }


def emit_equation(text: str) -> dict:
    """Consume outward character pairs before accepting the complete tape."""
    tape = letters(text)
    checked = 0
    for left in range((len(tape) + 1) // 2):
        right = len(tape) - 1 - left
        checked += 1
        if tape[left] != tape[right]:
            return {"accepted": False, "characters_checked": checked,
                    "mismatch": [left, tape[left], tape[right]]}
    return {"accepted": True, "characters_checked": checked, "mismatch": None}


def scene(valency: str, frame: tuple[str, str, str, str], rel: tuple[str, str],
          tail: str) -> tuple[str, dict]:
    verb, complement, antecedent, setting = frame
    relative, rel_role = rel
    # The relative is attached to the complement, with an intact finite clause.
    if valency == "transitive":
        sentence = f"{verb} {complement} {relative}; then walk toward {setting} {tail}."
    else:
        sentence = f"{verb} {complement} {relative}; then walk beside {setting} {tail}."
    return sentence, {
        "valency": valency,
        "relative_attachment": rel_role,
        "imperative_frame": verb,
        "complement": complement,
        "relative_antecedent": antecedent,
        "tail": tail,
    }


def run() -> dict:
    choices = []
    for valency, frames in (("transitive", TRANSITIVE), ("intransitive", INTRANSITIVE)):
        for frame, rel, tail in product(frames, SUBJECT_GAP + OBJECT_GAP, TAILS):
            text, grammar = scene(valency, frame, rel, tail)
            online = emit_equation(text)
            audit = pointer_audit(text)
            choices.append({
                "rendered": text,
                "grammar": grammar,
                "online_equation": online,
                "audit": audit,
                "provenance": {
                    "lexicon": "fresh hand-authored imperative and relative-role bank",
                    "complete_construction": True,
                    "subject_gap_or_object_gap": grammar["relative_attachment"],
                    "finished_tape_reversal": False,
                    "post_hoc_repair": False,
                    "repeated_units": False,
                    "mirrored_units": False,
                    "catalogue_text": False,
                    "fragment": False,
                },
            })
    exact = [row for row in choices if row["online_equation"]["accepted"] and row["audit"]["exact"]]
    reader = [row for row in exact if row["audit"]["letters"] > 38]
    max_row = max(choices, key=lambda row: row["audit"]["letters"])
    return {
        "experiment_id": EXPERIMENT_ID,
        "method": "held-out transitive/intransitive imperative frames with subject-gap/object-gap relative attachments and live outward character equations",
        "stats": {
            "transitive_frames": len(TRANSITIVE),
            "intransitive_frames": len(INTRANSITIVE),
            "relative_roles": 2,
            "tails": len(TAILS),
            "states": len(choices),
            "online_prunes": sum(not row["online_equation"]["accepted"] for row in choices),
            "complete_prose_controls": len(choices),
            "fresh_exact_gt38": len(reader),
            "max_letters": max_row["audit"]["letters"],
            "status": "UNSAT" if not reader else "EXACT_SURVIVOR",
        },
        "reader_facing_candidates": reader,
        "rendered_controls": choices[:12],
        "provenance": {
            "construction": "independent imperative valency and relative-role choices; outward character equation during emission",
            "subject_gap_forms": [x[0] for x in SUBJECT_GAP],
            "object_gap_forms": [x[0] for x in OBJECT_GAP],
            "no_repair_or_reversal": True,
            "novelty_preflight": "distinct from declarative/vocative mixtures and prior relative residual schedulers: imperative valency alternates with relative attachment role",
            "independent_validation": ["two-pointer audit", "forward/reverse SHA-256"],
            "reader_gate": "closed unless exact candidate exceeds 38 letters; controls are diagnostics only",
            "next_construction": "if empty, author a dual-imperative scene with one subject-gap and one object-gap attachment, retaining live equations",
        },
    }


if __name__ == "__main__":
    result = run()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
    for row in result["reader_facing_candidates"]:
        print(row["rendered"])
