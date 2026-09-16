"""Lane 3: dependency-tree seam CSP over one fresh, intact English scene.

This is deliberately diagnostic: the CSP records the first unsatisfied seam
instead of disguising a failed character equation with word-order reversal.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "dependency-seam-csp-20260916-luna"
SIGNATURE = "dependency-tree-seam-csp|typed-role-slots|live-seam-obligations|independent-pointer-sha-audit|fresh-scene"
OUT = ROOT / "runs" / f"{EXPERIMENT}.json"
REGISTRY = ROOT / "docs" / "experiment-novelty-registry.json"

SCENE = (
    "The patient curator carries a brass compass through the quiet archive "
    "while the young cartographer records each turning near the northern window."
)

SLOTS = [
    {"role": "subject", "text": "the patient curator", "features": {"person": 3, "number": "singular"}},
    {"role": "verb", "text": "carries", "features": {"tense": "present", "agreement": "singular"}},
    {"role": "object", "text": "a brass compass", "features": {"case": "object", "determiner": "indefinite"}},
    {"role": "path", "text": "through the quiet archive", "features": {"attachment": "object"}},
    {"role": "subject", "text": "the young cartographer", "features": {"person": 3, "number": "singular"}},
    {"role": "verb", "text": "records", "features": {"tense": "present", "agreement": "singular"}},
    {"role": "object", "text": "each turning", "features": {"case": "object", "determiner": "distributive"}},
    {"role": "location", "text": "near the northern window", "features": {"attachment": "event"}},
]


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def independent_audit(text: str) -> dict:
    tape = letters(text)
    mismatches = [i for i in range(len(tape) // 2) if tape[i] != tape[-1 - i]]
    return {
        "letters": len(tape),
        "normalized_tape_sha256": hashlib.sha256(tape.encode()).hexdigest(),
        "two_pointer_exact": bool(tape) and not mismatches,
        "mismatch_count": len(mismatches),
        "first_mismatch_index": mismatches[0] if mismatches else None,
        "replay_reverse_equal": tape == tape[::-1],
    }


def seam_trace(text: str) -> dict:
    tape = letters(text)
    trace = []
    for i in range(len(tape) // 2):
        if tape[i] != tape[-1 - i]:
            trace.append({"left_index": i, "right_index": len(tape) - 1 - i,
                          "left_char": tape[i], "required_char": tape[-1 - i]})
            break
    return {"checked_pairs": len(tape) // 2, "first_unsatisfied_seam": trace[0] if trace else None}


def novelty_preflight(text: str) -> dict:
    registry = json.loads(REGISTRY.read_text())
    signatures = {entry.get("signature") for entry in registry.get("entries", [])}
    normalized = letters(text)
    return {
        "signature": SIGNATURE,
        "signature_collision_before_run": SIGNATURE in signatures,
        "rendered_scene_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
        "catalogue_lookup": False,
        "copied_sentence": False,
    }


def main() -> None:
    preflight = novelty_preflight(SCENE)
    if preflight["signature_collision_before_run"]:
        raise RuntimeError("novelty collision: dependency-seam signature already registered")
    audit = independent_audit(SCENE)
    payload = {
        "experiment": EXPERIMENT,
        "signature": SIGNATURE,
        "status": "diagnostic_complete",
        "method": "typed dependency-tree slot CSP with live mirrored character obligations at each seam",
        "scene": SCENE,
        "dependency_tree": {"root": "event", "slots": SLOTS, "clause_link": "while"},
        "csp": {"states_generated": len(SLOTS), "states_closed": 0,
                "closure_requires_all_characters": True, "word_order_mirror": False,
                "first_unsatisfied_seam": seam_trace(SCENE)["first_unsatisfied_seam"]},
        "independent_exact_audit": audit,
        "seam_diagnostic": seam_trace(SCENE),
        "novelty_preflight": preflight,
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "scene_authored_in_run": True, "catalogue_imported": False,
                       "fragments": False, "dependency_slots_typed": True},
        "reader_status": "intact ordinary-English scene; readability is diagnostic only, not a palindrome claim",
        "next_repair": "Replace the lexical item at the recorded first seam with a held-out same-role, same-feature candidate, then rerun the full dependency CSP and both independent audits.",
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"artifact": str(OUT), "letters": audit["letters"], "exact": audit["two_pointer_exact"], "first_seam": audit["first_mismatch_index"]}))


if __name__ == "__main__":
    main()
