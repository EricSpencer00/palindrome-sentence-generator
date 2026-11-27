#!/usr/bin/env python3
"""Repunctuate one incomplete clause in the exact 752-letter rendering.

The letter tape is frozen. This is a presentation/grammar edit, not a length
search, a new normalized palindrome, or a readability evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "runs/repair-736-center-lexical-event-path-20260923.json"
OLD = "Nadia delivers maps. Leon. Ari sees Leon."
NEW = "Nadia delivers maps, Leon. Ari sees Leon."
EXPECTED_TAPE_SHA256 = "109c758ecfa0c0a2049c1dfb925d3820eb453a92836c0835d80f5106f222de3c"


def normalize(text: str) -> str:
    return re.sub(r"[^A-Za-z]", "", text).lower()


def exact(text: str) -> tuple[bool, int | None]:
    tape = normalize(text)
    for left in range(len(tape) // 2):
        right = len(tape) - left - 1
        if tape[left] != tape[right]:
            return False, left
    return True, None


def build() -> dict:
    source_doc = json.loads(PARENT.read_text())
    parent = source_doc["candidate"]["rendered"]
    parent_tape = normalize(parent)
    if source_doc["candidate"]["id"] != "568-lineage-central-event-path-752":
        raise ValueError("parent id changed")
    if len(parent_tape) != 752 or hashlib.sha256(parent_tape.encode()).hexdigest() != EXPECTED_TAPE_SHA256:
        raise ValueError("pinned parent tape changed")
    if parent.count(OLD) != 1:
        raise ValueError("expected exactly one frozen punctuation seam")
    if exact(parent) != (True, None):
        raise ValueError("pinned parent is no longer exact")

    child = parent.replace(OLD, NEW, 1)
    child_tape = normalize(child)
    is_exact, mismatch = exact(child)
    child_hash = hashlib.sha256(child_tape.encode()).hexdigest()
    if child_tape != parent_tape:
        raise ValueError("punctuation edit unexpectedly changed the letter tape")
    if not is_exact:
        raise ValueError(f"child is not exact at offset {mismatch}")

    return {
        "experiment_id": "center-vocative-repunctuation-20261008",
        "status": "same_tape_grammar_repair_exactness_preserved",
        "question": "Can an inherited one-word fragment be joined as a grammatical vocative without changing the exact letter tape?",
        "plan": {
            "decision": "Test a presentation-level grammar repair at one measured malformed seam in the verified 752 rendering.",
            "acceptance_gate": "Exactly one frozen substring changes; normalized tape and length remain identical; independent opposing-character check passes; revised syntax reads as a finite clause with direct address.",
            "baseline": "Verified 752-letter central-event-path rendering.",
            "falsifier": "Any normalized-letter change, mismatch, wrong seam, or syntax that does not support a vocative rejects the repair.",
            "fixed_conditions": ["one punctuation substitution", "no lexical substitution", "no search or model scoring"],
            "stop_rule": "One localized punctuation repair only.",
            "promotion_rule": "May be retained as an edited rendering of the same exact candidate; does not establish whole-text readability."
        },
        "parent": {
            "artifact": "runs/repair-736-center-lexical-event-path-20260923.json",
            "candidate_id": "568-lineage-central-event-path-752",
            "normalized_letters": len(parent_tape),
            "normalized_sha256": EXPECTED_TAPE_SHA256
        },
        "edit": {
            "old": OLD,
            "new": NEW,
            "operation": "replace the period after 'maps' with a comma so 'Leon' is a direct-address vocative",
            "local_before": parent[parent.index(OLD) - 35 : parent.index(OLD) + len(OLD) + 35],
            "local_after": child[child.index(NEW) - 35 : child.index(NEW) + len(NEW) + 35],
            "characters_changed": "one punctuation mark",
            "letter_tape_changed": False,
            "grammar_note": "'Nadia delivers maps, Leon.' is a finite clause with a terminal vocative; the next sentence refers to Leon again. This local reading is a syntax judgment by the author, not a reader rating."
        },
        "candidate": {
            "id": "568-lineage-central-event-path-752-vocative-edit",
            "rendered": child,
            "normalized_letters": len(child_tape),
            "growth_over_parent": 0,
            "normalized_tape_sha256": child_hash
        },
        "independent_audit": {
            "normalizer": "remove non-ASCII letters and lowercase",
            "two_pointer_exact": is_exact,
            "first_mismatch": mismatch,
            "forward_sha256": child_hash,
            "reverse_sha256": hashlib.sha256(child_tape[::-1].encode()).hexdigest(),
            "hashes_equal": child_hash == hashlib.sha256(child_tape[::-1].encode()).hexdigest(),
            "normalized_tape_equal_to_parent": child_tape == parent_tape
        },
        "limitations": {
            "length_frontier_advanced": False,
            "full_text_readability": "not assessed",
            "human_ratings": "none",
            "search_scope": "one authored punctuation repair; no remote compute used",
            "public_claim": False
        },
        "provenance": {
            "parent_file_sha256": hashlib.sha256(PARENT.read_bytes()).hexdigest(),
            "source_script": "experiments/center_vocative_repunctuation_20261008.py"
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "runs/center-vocative-repunctuation-20261008.json")
    args = parser.parse_args()
    result = build()
    result["provenance"]["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "status": result["status"],
        "letters": result["candidate"]["normalized_letters"],
        "growth": result["candidate"]["growth_over_parent"],
        "exact": result["independent_audit"]["two_pointer_exact"],
        "sha256": result["candidate"]["normalized_tape_sha256"],
        "out": str(args.out)
    }, indent=2))


if __name__ == "__main__":
    main()
