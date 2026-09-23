#!/usr/bin/env python3
"""Preserve and reject an exact 568-lineage child with a phrasewise seam.

The trial is useful negative evidence: its local clauses close exactly and
grow the pinned tape, but a shared reflected word boundary splits the local
equation into two smaller reverse phrase pairs.  It is not a candidate or a
readability claim.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import (
    normalize_letters,
    phrasewise_reverse_boundary_offsets,
)
from llm_palindrome.validator import is_palindrome


PARENT_PATH = "runs/incumbent-560-outer-causal-scene-20261002.json"
PARENT_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
OUTPUT_PATH = "runs/incumbent-568-phrasewise-equation-audit-20260923.json"
LEFT_SPAN = (148, 163)
RIGHT_SPAN = (405, 420)
LEFT_PHRASE = "Reda Mar diapered Elena."
RIGHT_PHRASE = "Ane Lede repaid Rama Der."


def independent_tape(text: str) -> str:
    """Normalize independently of the project's validator and admission code."""
    return "".join(character.lower() for character in text
                   if character.isascii() and character.isalpha())


def replace_normalized_span(text: str, start: int, end: int, replacement: str) -> str:
    offsets = [index for index, char in enumerate(text)
               if char.isascii() and char.isalpha()]
    if not (0 <= start < end < len(offsets)):
        raise ValueError("span must be a nonempty interior normalized-letter range")
    return text[:offsets[start]] + replacement + " " + text[offsets[end]:]


def make_candidate(parent_text: str) -> str:
    candidate = parent_text
    # Replacing the higher span first keeps the parent's lower raw offsets valid.
    for (start, end), phrase in sorted(
        ((LEFT_SPAN, LEFT_PHRASE), (RIGHT_SPAN, RIGHT_PHRASE)),
        key=lambda item: item[0][0], reverse=True,
    ):
        candidate = replace_normalized_span(candidate, start, end, phrase)
    return candidate


def run() -> dict[str, object]:
    parent_artifact = json.loads((ROOT / PARENT_PATH).read_text())
    parent_text = parent_artifact["rows"][0]["rendered"]
    parent_tape = independent_tape(parent_text)
    parent_sha = hashlib.sha256(parent_tape.encode("ascii")).hexdigest()
    if len(parent_tape) != 568 or parent_sha != PARENT_SHA256:
        raise AssertionError("pinned parent changed")

    candidate = make_candidate(parent_text)
    tape = independent_tape(candidate)
    local_left = independent_tape(LEFT_PHRASE)
    local_right = independent_tape(RIGHT_PHRASE)
    if not is_palindrome(candidate) or tape != tape[::-1]:
        raise AssertionError("candidate must be independently exact for this audit")

    boundary_offsets = phrasewise_reverse_boundary_offsets(LEFT_PHRASE, RIGHT_PHRASE)
    split = boundary_offsets[0]
    left_prefix, left_suffix = local_left[:split], local_left[split:]
    right_suffix, right_prefix = local_right[-split:], local_right[:-split]
    if left_prefix != right_suffix[::-1] or left_suffix != right_prefix[::-1]:
        raise AssertionError("reported boundary must expose two exact reverse chunks")

    return {
        "experiment_id": "incumbent-568-phrasewise-equation-audit-20260923",
        "status": "exact_audit_only_rejected_phrasewise_local_equation",
        "method": "bounded authored event-clause substitution at a clean mirrored 568 seam, followed by reflected word-boundary factorization audit",
        "parent": {
            "artifact": PARENT_PATH,
            "normalized_letters": len(parent_tape),
            "sha256": parent_sha,
            "normalized_replacement_spans": [list(LEFT_SPAN), list(RIGHT_SPAN)],
        },
        "local_equation": {
            "left_rendered": LEFT_PHRASE,
            "right_rendered": RIGHT_PHRASE,
            "left_tape": local_left,
            "right_tape": local_right,
            "exact_reverse_equation": local_left == local_right[::-1],
            "left_internal_word_boundaries": [4, 7, 15],
            "reflected_right_internal_word_boundaries": [3, 7, 13, 17],
            "shared_reflected_boundaries": list(boundary_offsets),
            "factorization_at_offset": split,
            "reverse_chunks": [
                {"left": left_prefix, "right": right_suffix,
                 "right_reversed": right_suffix[::-1]},
                {"left": left_suffix, "right": right_prefix,
                 "right_reversed": right_prefix[::-1]},
            ],
            "anti_shortcut_pass": not boundary_offsets,
        },
        "candidate": {
            "rendered": candidate,
            "letters": len(tape),
            "sha256": hashlib.sha256(tape.encode("ascii")).hexdigest(),
            "project_validator_exact": is_palindrome(candidate),
            "independent_two_pointer_exact": all(
                tape[index] == tape[-index - 1]
                for index in range(len(tape) // 2)
            ),
            "growth_over_parent": len(tape) - len(parent_tape),
            "reader_evidence": False,
            "readability_status": "not reader-tested; the invented-looking proper-name cluster is not claimed as readable prose",
            "admission_status": "rejected: local exact equation factors at a shared noncentral reflected word boundary",
        },
        "provenance": {
            "source": "bounded role-preserving clause substitution proposed in the anaphoric/inflection lane",
            "borrowed_text": False,
            "finished_tape_reversed_to_construct": False,
            "reader_claim": False,
            "new_insertions": [LEFT_PHRASE, RIGHT_PHRASE],
        },
        "next_action": "keep the boundary test; abandon diapered/repaid and preregister a different auxiliary-participle versus finite-verb/object-complement topology at the same clean parent seam",
    }


if __name__ == "__main__":
    result = run()
    (ROOT / OUTPUT_PATH).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "artifact": OUTPUT_PATH,
        "letters": result["candidate"]["letters"],
        "sha256": result["candidate"]["sha256"],
        "exact": result["candidate"]["project_validator_exact"],
        "local_boundary_rejection": result["local_equation"]["shared_reflected_boundaries"],
    }, indent=2))
