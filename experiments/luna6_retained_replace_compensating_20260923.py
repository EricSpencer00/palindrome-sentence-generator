#!/usr/bin/env python3
"""One retained-span replacement plus a distinct sentence deletion on pinned 568."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PARENT = ROOT / "runs/incumbent-560-outer-causal-scene-20261002.json"
OUTPUT = ROOT / "runs/luna6-retained-replace-compensating-20260923.json"
EXPECTED = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"


def tape(text: str) -> str:
    return "".join(c.lower() for c in text if c.isalpha())


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def outside_in(text: str) -> dict:
    letters = tape(text)
    mismatch = next(
        ((i, letters[i], letters[-1 - i]) for i in range(len(letters) // 2)
         if letters[i] != letters[-1 - i]),
        None,
    )
    return {
        "letters": len(letters),
        "exact": mismatch is None,
        "first_mismatch": (
            {"offset": mismatch[0], "left": mismatch[1], "right": mismatch[2],
             "right_offset": len(letters) - 1 - mismatch[0]}
            if mismatch else None
        ),
    }


def project_audit(text: str) -> dict:
    from llm_palindrome.validator import is_palindrome

    return {"exact": bool(is_palindrome(text))}


def main() -> None:
    parent_doc = json.loads(PARENT.read_text())
    parent = parent_doc["rows"][0]["rendered"]
    parent_tape = tape(parent)
    if digest(parent_tape) != EXPECTED:
        raise SystemExit("Pinned parent hash mismatch")
    if not outside_in(parent)["exact"] or not project_audit(parent)["exact"]:
        raise SystemExit("Pinned parent failed independent exactness checks")

    # One authored scene realization; not a phrase bank or a geometry sweep.
    # Replace the whole sentence “Mara saw God.” by a fresh coherent sentence.
    # Delete the separate whole sentence “Leon.” between two complete clauses.
    rendered = parent.replace(
        "Mara saw God.", "Mara carried silver keys home."
    ).replace("Leon. Ari delivers maps.", "Ari delivers maps.")
    if rendered == parent:
        raise SystemExit("Expected source spans not found verbatim")
    child_tape = tape(rendered)

    # Record the residual the replacement must satisfy after the later deletion.
    # Wildcards occupy the new replacement's 25 letters; their opposite cells
    # are all retained parent material, so these characters are forced.
    start, new_width = 194, len(tape("Mara carried silver keys home"))
    old_start, old_width = 194, len(tape("Mara saw God"))
    delete_start, delete_width = 265, len(tape("Leon"))
    shifted_delete_start = delete_start + (new_width - old_width)
    wildcard = (
        parent_tape[:start]
        + "?" * new_width
        + parent_tape[old_start + old_width:delete_start]
        + parent_tape[delete_start + delete_width:]
    )
    forced = []
    for pos in range(start, start + new_width):
        opposite = len(wildcard) - 1 - pos
        forced.append(wildcard[opposite])
    forced_tape = "".join(forced)
    proposed_tape = tape("Mara carried silver keys home")
    if len(wildcard) != len(child_tape):
        raise SystemExit("Residual model length does not match rendered child")

    # Surface token audit applies to the newly authored clause only. Existing
    # parent repair debt is preserved, not represented as new material.
    new_tokens = ["mara", "carried", "silver", "keys", "home"]
    reverse_pairs = [
        [a, b] for i, a in enumerate(new_tokens) for b in new_tokens[i + 1:]
        if a[::-1] == b
    ]
    self_palindromes = [w for w in new_tokens if w == w[::-1]]
    repeated_tokens = sorted({w for w in new_tokens if new_tokens.count(w) > 1})
    child_audit = outside_in(rendered)
    if child_audit["exact"]:
        raise SystemExit("Unexpected closure: inspect before recording this diagnostic")

    record = {
        "experiment_id": "luna6-retained-replace-compensating-20260923",
        "status": "rejected_exactness_residual_obstruction",
        "method": "replace one complete retained sentence with a fresh scene sentence, then delete one distinct complete sentence; carry the reflection map through both edits",
        "novelty_preflight": {
            "parent": "runs/incumbent-560-outer-causal-scene-20261002.json",
            "parent_sha256": EXPECTED,
            "prior_nonmirror_insertion_diagnostic": "runs/luna6-nonmirror-shift-seam-20260923.json",
            "distinct_from": [
                "reflected-cut identity insertion",
                "single replacement-only seam edits",
                "event-cycle growth",
                "16/543, 64/504, and 178/390 insertion geometries",
            ],
            "preflight_result": "No exact geometry was reused: this run replaces the sentence at normalized [194,204) and deletes a different complete sentence at [265,269); the second cut is shifted to [280,284) after the replacement expansion.",
        },
        "provenance": {
            "parent_artifact": str(PARENT.relative_to(ROOT)),
            "parent_letters": len(parent_tape),
            "parent_normalized_sha256": digest(parent_tape),
            "parent_outside_in": outside_in(parent),
            "parent_project_validator": project_audit(parent),
            "replacement_source": "Mara saw God.",
            "replacement_source_normalized_span": [194, 204],
            "replacement_text": "Mara carried silver keys home.",
            "replacement_normalized_letters": new_width,
            "deletion_source": "Leon.",
            "deletion_parent_normalized_span": [delete_start, delete_start + delete_width],
            "deletion_postreplacement_normalized_span": [shifted_delete_start, shifted_delete_start + delete_width],
            "net_growth": new_width - old_width - delete_width,
        },
        "rendered": rendered,
        "candidate": {
            "letters": len(child_tape),
            "normalized_sha256": digest(child_tape),
            "outside_in": child_audit,
            "project_validator": project_audit(rendered),
            "exact": False,
        },
        "seam_ledger": {
            "parent": {"letters": 568, "state": "exact; no live residual"},
            "after_replacement_before_deletion": {
                "letters": len(parent_tape) + new_width - old_width,
                "first_mismatch": {"offset": 198, "left": "c", "right": "s", "right_offset": 384},
            },
            "after_compensating_deletion": {
                "letters": len(child_tape),
                "first_mismatch": child_audit["first_mismatch"],
                "residual_at_replacement": {
                    "normalized_span": [start, start + new_width],
                    "forced_by_unchanged_opposite_material": forced_tape,
                    "proposed": proposed_tape,
                    "first_local_conflict": {
                        "candidate_offset": 198,
                        "proposed": proposed_tape[4],
                        "required": forced_tape[4],
                        "opposite_offset": len(wildcard) - 1 - 198,
                    },
                    "fully_forced": all(c != "?" for c in forced),
                },
            },
        },
        "boundary_audit": {
            "new_clause_tokens": new_tokens,
            "whole_token_reversal_pairs_within_new_clause": reverse_pairs,
            "self_palindromic_tokens": self_palindromes,
            "repeated_tokens": repeated_tokens,
            "shortcut_free_new_clause": not (reverse_pairs or self_palindromes or repeated_tokens),
            "note": "The trial fails exactness before any readability claim; inherited parent text remains unchanged outside these two edits.",
        },
        "readability": {
            "status": "not_reader_evaluated",
            "reason": "The exactness equation fails at the first new-scene seam; this is not eligible for a reader study.",
        },
        "next_operator": "Move the compensating deletion into the reflection footprint of the replaced sentence so that at least one new clause-internal span is variable-to-variable rather than fully forced by retained text; preflight that distinct geometry first, then author the clause under the resulting live residual. Do not swap words in this rejected sentence or reuse [194,204)/[265,269).",
    }
    OUTPUT.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "artifact": str(OUTPUT), "length": len(child_tape),
        "sha256": digest(child_tape), "first_mismatch": child_audit["first_mismatch"],
        "forced_replacement_residual": forced_tape,
    }, indent=2))


if __name__ == "__main__":
    main()
