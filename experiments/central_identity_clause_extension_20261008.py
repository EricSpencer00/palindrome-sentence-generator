#!/usr/bin/env python3
"""Authored center seam extension of the independently exact 752-letter row.

This is one frozen construction, not a search or readability evaluation.  It
adds two finite clauses at the center of the pinned tape.  Their normalized
letter tapes are exact reverses, and they share a subject/complement identity
interpretation.  All parent characters remain in their original order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARENT_PATH = ROOT / "runs/repair-736-center-lexical-event-path-20260923.json"
EXPECTED_PARENT_ID = "568-lineage-central-event-path-752"
EXPECTED_PARENT_LENGTH = 752
EXPECTED_PARENT_SHA256 = "109c758ecfa0c0a2049c1dfb925d3820eb453a92836c0835d80f5106f222de3c"
LEFT_CLAUSE = "Noel saw Leon."
RIGHT_CLAUSE = "Noel was Leon."
LEFT_SEAM_CONTEXT = "Leon stops Ira. "
RIGHT_SEAM_CONTEXT = "Ari spots Noel."


def normalize(text: str) -> str:
    return re.sub(r"[^A-Za-z]", "", text).lower()


def two_pointer_exact(text: str) -> tuple[bool, int | None]:
    tape = normalize(text)
    for i in range(len(tape) // 2):
        j = len(tape) - 1 - i
        if tape[i] != tape[j]:
            return False, i
    return True, None


def cut_for_normalized_offset(text: str, offset: int) -> int:
    count = 0
    for raw, char in enumerate(text):
        if char.isascii() and char.isalpha():
            if count == offset:
                return raw
            count += 1
    if count == offset:
        return len(text)
    raise ValueError(f"offset {offset} exceeds normalized tape length {count}")


def build() -> dict:
    parent_doc = json.loads(PARENT_PATH.read_text())
    candidate = parent_doc["candidate"]
    parent = candidate["rendered"]
    parent_tape = normalize(parent)
    if candidate["id"] != EXPECTED_PARENT_ID:
        raise ValueError("pinned parent id drift")
    if len(parent_tape) != EXPECTED_PARENT_LENGTH:
        raise ValueError("pinned parent length drift")
    if hashlib.sha256(parent_tape.encode()).hexdigest() != EXPECTED_PARENT_SHA256:
        raise ValueError("pinned parent tape hash drift")
    parent_exact, mismatch = two_pointer_exact(parent)
    if not parent_exact:
        raise ValueError(f"parent is not exact at normalized offset {mismatch}")

    target = len(parent_tape) // 2
    cut = cut_for_normalized_offset(parent, target)
    if parent[cut - len(LEFT_SEAM_CONTEXT) : cut] != LEFT_SEAM_CONTEXT:
        raise ValueError("frozen left center context is not at the parent half cut")
    if parent[cut : cut + len(RIGHT_SEAM_CONTEXT)] != RIGHT_SEAM_CONTEXT:
        raise ValueError("frozen center context is not at the parent half cut")

    left_tape = normalize(LEFT_CLAUSE)
    right_tape = normalize(RIGHT_CLAUSE)
    if left_tape != right_tape[::-1]:
        raise ValueError("authored local clause equation did not close")
    inserted = LEFT_CLAUSE + " " + RIGHT_CLAUSE + " "
    child = parent[:cut] + inserted + parent[cut:]
    child_tape = normalize(child)
    exact, first_mismatch = two_pointer_exact(child)
    child_hash = hashlib.sha256(child_tape.encode()).hexdigest()

    # Verify the operation itself, including byte-for-byte preservation of the
    # parent outside the one insertion point.
    recovered = child[:cut] + child[cut + len(inserted) :]
    if recovered != parent:
        raise ValueError("parent was not retained byte-for-byte around insertion")
    if not child_tape.startswith(parent_tape[:target]):
        raise ValueError("left retained prefix changed")

    return {
        "experiment_id": "central-identity-clause-extension-20261008",
        "status": "rejected_exact_extension_novelty_and_shortcut_gate",
        "question": "Can one connected, non-SVO identity clause pair extend the verified 752-letter parent at its center while preserving the parent tape?",
        "plan": {
            "decision": "Determine whether a different clause family can add an exact center block with retained context.",
            "acceptance_gate": "In addition to parent retention, positive growth, local equation, exactness, and finite-clause interpretation, the inserted clauses must be novel in the tracked archive and must not rely on direct reversed content-word tokens.",
            "baseline": "Verified 752-letter parent and its exact center split.",
            "fixed_conditions": ["one deterministic authored insertion", "ASCII-letter lowercase normalization", "no corpus or model scoring", "no parameter tuning"],
            "falsifier": "Any parent drift, seam drift, local equation mismatch, child mismatch, or failure to retain the original tape aborts the construction.",
            "stop_rule": "One authored candidate only; no lexical expansion after observing the result.",
            "promotion_rule": "Keep as an exploratory exact construction only; no readability claim without blinded human ratings."
        },
        "parent": {
            "artifact": "runs/repair-736-center-lexical-event-path-20260923.json",
            "candidate_id": EXPECTED_PARENT_ID,
            "letters": EXPECTED_PARENT_LENGTH,
            "normalized_sha256": EXPECTED_PARENT_SHA256,
            "independent_two_pointer_exact": parent_exact
        },
        "method": "Diagnostic authored insertion at the verified center cut: a transitive-past clause followed by a copular-past identity clause. This is preserved as a rejected probe, not promoted as a new construction.",
        "seam": {
            "normalized_offset": target,
            "raw_offset": cut,
            "parent_context_before": parent[max(0, cut - 80) : cut],
            "parent_context_after": parent[cut : cut + 80],
            "frozen_context_witness": LEFT_SEAM_CONTEXT + "|" + RIGHT_SEAM_CONTEXT,
            "outside_insertion_retained_byte_for_byte": recovered == parent
        },
        "local_equation": {
            "left_clause": LEFT_CLAUSE,
            "left_tape": left_tape,
            "right_clause": RIGHT_CLAUSE,
            "right_tape": right_tape,
            "reverse_right_tape": right_tape[::-1],
            "exact": left_tape == right_tape[::-1],
            "letters_per_clause": len(left_tape),
            "interpretation": "A discovery followed by an identity disclosure (Noel is Leon); this reading is an authored interpretation, not a reader judgment."
        },
        "rejection": {
            "reasons": [
                "The exact clause 'Noel saw Leon' already occurs in tracked exploratory artifacts, including runs/semantic-event-argument-csp-20260918.json.",
                "Noel and Leon are direct reversed-name tokens; the inserted identity pair therefore relies on a lexical reversal shortcut."
            ],
            "effect": "Exactness is retained as a diagnostic, but this does not meet the novelty/shortcut gate and is not a promoted candidate."
        },
        "candidate": {
            "id": "568-lineage-central-identity-clause-774",
            "rendered": child,
            "letters": len(child_tape),
            "growth_over_parent": len(child_tape) - len(parent_tape),
            "normalized_tape_sha256": child_hash
        },
        "independent_audit": {
            "normalizer": "remove non-ASCII letters and lowercase",
            "normalized_letters": len(child_tape),
            "two_pointer_exact": exact,
            "first_mismatch": first_mismatch,
            "forward_sha256": child_hash,
            "reverse_sha256": hashlib.sha256(child_tape[::-1].encode()).hexdigest(),
            "hashes_equal": child_hash == hashlib.sha256(child_tape[::-1].encode()).hexdigest(),
            "parent_recovered_byte_for_byte": recovered == parent
        },
        "limitations": {
            "full_text_readability": "not assessed",
            "human_ratings": "none",
            "meaning_scope": "only the inserted two-clause unit has an authored identity reading; inherited rough discourse and repetition remain",
            "search_scope": "not a search run; no remote compute used",
            "public_claim": False
        },
        "source": {
            "script": "experiments/central_identity_clause_extension_20261008.py",
            "parent_file_sha256": hashlib.sha256(PARENT_PATH.read_bytes()).hexdigest()
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "runs/central-identity-clause-extension-20261008.json")
    args = parser.parse_args()
    result = build()
    result["source"]["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "status": result["status"],
        "letters": result["candidate"]["letters"],
        "growth": result["candidate"]["growth_over_parent"],
        "exact": result["independent_audit"]["two_pointer_exact"],
        "sha256": result["candidate"]["normalized_tape_sha256"],
        "out": str(args.out)
    }, indent=2))


if __name__ == "__main__":
    main()
