"""Record an exact but novelty-rejected asymmetric residual insertion.

The experiment inserts ``Noel sees`` and ``Leon sees`` at different cuts in
the exact 568-letter incumbent.  Its live residual equation closes exactly,
but repository-wide preflight found those clause stems in earlier generated
runs, so this row is retained only as rejected novelty evidence.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.validator import is_palindrome, normalize


PARENT = ROOT / "runs" / "incumbent-560-outer-causal-scene-20261002.json"
OUT = ROOT / "runs" / "incumbent-568-asymmetric-residual-insertion-20260922.json"
PARENT_ID = "outer-causal-scene-568-working-incumbent"
PARENT_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
CHILD_SHA256 = "ba1bd287e0cacff6eb4a7a6891fd324ddc8500dd14c0a481cba573c321afd331"
LEFT_CUT = 151
RIGHT_CUT = 413
LEFT_INSERTION = "noelsees"
RIGHT_INSERTION = "leonsees"


def independent_tape(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def raw_boundary_after_letters(text: str, count: int) -> int:
    seen = 0
    for index, char in enumerate(text):
        if char.isascii() and char.isalpha():
            seen += 1
            if seen == count:
                return index + 1
    raise ValueError(f"surface has fewer than {count} ASCII letters")


def independent_two_pointer(tape: str) -> bool:
    left, right = 0, len(tape) - 1
    while left < right:
        if tape[left] != tape[right]:
            return False
        left += 1
        right -= 1
    return bool(tape)


def audit(text: str) -> dict[str, object]:
    tape = independent_tape(text)
    forward = hashlib.sha256(tape.encode("ascii")).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode("ascii")).hexdigest()
    return {
        "letters": len(tape),
        "two_pointer_exact": independent_two_pointer(tape),
        "project_validator_exact": bool(is_palindrome(text)),
        "project_normalizer_agrees": tape == normalize(text),
        "sha256_forward": forward,
        "sha256_reverse": reverse,
        "sha_equal": forward == reverse,
    }


def load_parent() -> tuple[str, str]:
    payload = json.loads(PARENT.read_text())
    row = next(item for item in payload["rows"] if item["id"] == PARENT_ID)
    rendered = str(row["rendered"])
    tape = independent_tape(rendered)
    assert row["audit"]["letters"] == len(tape) == 568
    assert row["audit"]["sha256_forward"] == PARENT_SHA256
    assert tape == tape[::-1]
    return rendered, tape


def build_payload() -> dict[str, object]:
    parent_rendered, parent_tape = load_parent()
    assert parent_tape[LEFT_CUT : LEFT_CUT + 4] == "noel"
    assert parent_tape[RIGHT_CUT : RIGHT_CUT + 4] == "leon"

    left_raw = raw_boundary_after_letters(parent_rendered, LEFT_CUT)
    right_raw = raw_boundary_after_letters(parent_rendered, RIGHT_CUT)
    # Insert at cuts measured on the unchanged parent; apply right-to-left so
    # the recorded raw offsets remain valid.
    rendered = parent_rendered[:right_raw] + " Leon sees" + parent_rendered[right_raw:]
    rendered = rendered[:left_raw] + " Noel sees" + rendered[left_raw:]

    # Turn the insertion joins into complete ordinary clauses.  These edits
    # touch punctuation and whitespace only; the letter tape is asserted below.
    rendered = rendered.replace(
        "Now Noel sees, Noel, did I live?",
        "Now Noel sees. Noel, did I live?",
        1,
    )
    rendered = rendered.replace(
        "Evil I did Leon sees, Leon won.",
        "Evil I did. Leon sees. Leon won.",
        1,
    )

    child_tape = independent_tape(rendered)
    expected_tape = (
        parent_tape[:LEFT_CUT]
        + LEFT_INSERTION
        + parent_tape[LEFT_CUT:RIGHT_CUT]
        + RIGHT_INSERTION
        + parent_tape[RIGHT_CUT:]
    )
    result_audit = audit(rendered)
    assert child_tape == expected_tape
    assert result_audit["letters"] == 584
    assert result_audit["sha256_forward"] == CHILD_SHA256
    assert all(
        result_audit[key]
        for key in (
            "two_pointer_exact",
            "project_validator_exact",
            "project_normalizer_agrees",
            "sha_equal",
        )
    )

    residual = "noel"
    left_emission = LEFT_INSERTION
    right_emission = RIGHT_INSERTION
    equation_left = left_emission + residual
    equation_right = residual + right_emission[::-1]
    assert equation_left == equation_right == "noelseesnoel"

    return {
        "experiment_id": "incumbent-568-asymmetric-residual-insertion-20260922",
        "method": "asymmetric off-center insertion with a live lexical residual",
        "status": "exact_but_novelty_rejected",
        "parent": {
            "artifact": str(PARENT.relative_to(ROOT)),
            "id": PARENT_ID,
            "letters": 568,
            "sha256": PARENT_SHA256,
        },
        "candidate": {
            "id": "asymmetric-noel-leon-residual-584",
            "rendered": rendered,
            "audit": result_audit,
            "letters_gained": 16,
            "sha256": CHILD_SHA256,
            "provenance": {
                "left_parent_cut": LEFT_CUT,
                "right_parent_cut": RIGHT_CUT,
                "left_added_clause": "Noel sees.",
                "right_added_clause": "Leon sees.",
                "new_full_clause_hits_in_parent": 0,
                "new_full_clause_hits_in_prior_experiment_files": True,
                "prior_collision_evidence": [
                    "experiments/incumbent_568_dual_seam_event_graft_20260922.py",
                    "runs/semantic-seam-frame-search-20260920.json",
                    "experiments/finite_predicate_complement_20260918.py",
                    "runs/human-scene-edge-lattice-20260918.json",
                ],
                "borrowed_or_catalogue_text": False,
                "finished_tape_reversal": False,
                "symmetric_wrapper": False,
                "punctuation_changed_letter_tape": False,
            },
            "live_residual": {
                "owner": "left insertion holds the four-letter noel obligation until right insertion consumes it",
                "residual": residual,
                "left_emission": left_emission,
                "right_emission": right_emission,
                "equation": "left_emission + residual = residual + reverse(right_emission)",
                "left_value": equation_left,
                "right_value": equation_right,
                "closed": True,
            },
            "repair_debt": {
                "inherited_proper_palindromic_spans": True,
                "inherited_repeated_scaffolding": True,
                "inherited_rough_prose": True,
                "human_readability_certified": False,
                "effect": "retain as exact growth evidence, not as a reader-certified result",
            },
        },
        "novelty_preflight": {
            "status": "rejected_exact_clause_reuse",
            "checked": [
                "runs/",
                "experiments/",
                "docs/",
                "data/",
            ],
            "exact_added_clause_collision": True,
            "colliding_phrases": ["Noel sees", "Leon sees"],
            "previous_symmetric_outer_shell_reuse": False,
            "new_dimension": "unequal insertion cuts coupled by an interior lexical residual",
        },
        "stats": {
            "exact_trials": 1,
            "novelty_admitted_children": 0,
            "longest_exact_but_rejected_trial": result_audit["letters"],
            "growth_over_parent": 16,
            "committed_character_contradictions": 0,
            "backtracks": 0,
        },
        "failed_parallel_probe": {
            "method": "central clause insertion at the incumbent's exact midpoint",
            "obstruction": "the parent midpoint bisects maps as map|s (mirrored s|pam); a clause insertion there strands a word fragment",
            "repair": "moved to the non-central [151, 413] cuts and carried the noel residual",
        },
        "next_repair": {
            "reason": "the exact event stems already occur in prior generated records",
            "operator": "replace the reused predicate with an unattested intransitive palindrome verb while retaining the same live noel residual",
            "target": "the inherited repeated 'A tub? He maps' shell",
            "gate": "retain 568 as the eligible parent; preserve 584 only as exact-but-duplicate evidence",
        },
    }


def main() -> None:
    payload = build_payload()
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
    print(payload["candidate"]["rendered"])


if __name__ == "__main__":
    main()
