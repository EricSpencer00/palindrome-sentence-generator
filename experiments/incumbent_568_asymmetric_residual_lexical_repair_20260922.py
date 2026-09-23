"""Repair a reused event predicate while preserving the 568 seam equation."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.incumbent_568_asymmetric_residual_insertion_20260922 import (
    LEFT_CUT,
    PARENT,
    PARENT_ID,
    PARENT_SHA256,
    RIGHT_CUT,
    audit,
    independent_tape,
    load_parent,
    raw_boundary_after_letters,
)


OUT = ROOT / "runs" / "incumbent-568-asymmetric-residual-solos-20260922.json"
PREDICATE = "solos"
LEFT_INSERTION = "noel" + PREDICATE
RIGHT_INSERTION = "leon" + PREDICATE
EXPECTED_SHA256 = "d7cf3db4224acc9042f7f4badd48dfbd69c56f0f554ed9d52f053ef5f197ab41"


def two_pointer(tape: str) -> bool:
    left, right = 0, len(tape) - 1
    while left < right:
        if tape[left] != tape[right]:
            return False
        left += 1
        right -= 1
    return bool(tape)


def build_payload() -> dict[str, object]:
    parent_rendered, parent_tape = load_parent()
    assert parent_tape[LEFT_CUT : LEFT_CUT + 4] == "noel"
    assert parent_tape[RIGHT_CUT : RIGHT_CUT + 4] == "leon"

    left_raw = raw_boundary_after_letters(parent_rendered, LEFT_CUT)
    right_raw = raw_boundary_after_letters(parent_rendered, RIGHT_CUT)
    rendered = parent_rendered[:right_raw] + " Leon solos" + parent_rendered[right_raw:]
    rendered = rendered[:left_raw] + " Noel solos" + rendered[left_raw:]
    before = rendered
    rendered = rendered.replace(
        "Now Noel solos, Noel, did I live?",
        "Now Noel solos. Noel, did I live?",
        1,
    )
    rendered = rendered.replace(
        "Evil I did Leon solos, Leon won.",
        "Evil I did. Leon solos. Leon won.",
        1,
    )
    assert rendered != before

    tape = independent_tape(rendered)
    expected_tape = (
        parent_tape[:LEFT_CUT]
        + LEFT_INSERTION
        + parent_tape[LEFT_CUT:RIGHT_CUT]
        + RIGHT_INSERTION
        + parent_tape[RIGHT_CUT:]
    )
    result_audit = audit(rendered)
    assert tape == expected_tape
    assert result_audit["letters"] == 586
    assert result_audit["sha256_forward"] == EXPECTED_SHA256
    assert two_pointer(tape)
    assert result_audit["project_validator_exact"] and result_audit["sha_equal"]

    residual = "noel"
    equation_left = LEFT_INSERTION + residual
    equation_right = residual + RIGHT_INSERTION[::-1]
    assert equation_left == equation_right == "noelsolosnoel"

    return {
        "experiment_id": "incumbent-568-asymmetric-residual-solos-20260922",
        "method": "asymmetric off-center lexical residual with a novelty-triggered intransitive-predicate repair",
        "status": "exact_growth_frontier_with_inherited_repair_debt",
        "parent": {
            "artifact": str(PARENT.relative_to(ROOT)),
            "id": PARENT_ID,
            "letters": 568,
            "sha256": PARENT_SHA256,
        },
        "candidate": {
            "id": "asymmetric-noel-leon-solos-586",
            "rendered": rendered,
            "audit": result_audit,
            "independent_outside_in_audit": {
                "letters": len(tape),
                "two_pointer_exact": two_pointer(tape),
                "forward_sha256": hashlib.sha256(tape.encode("ascii")).hexdigest(),
                "reverse_sha256": hashlib.sha256(tape[::-1].encode("ascii")).hexdigest(),
            },
            "growth_over_parent": 18,
            "provenance": {
                "left_parent_cut": LEFT_CUT,
                "right_parent_cut": RIGHT_CUT,
                "left_added_clause": "Noel solos.",
                "right_added_clause": "Leon solos.",
                "new_event_content": True,
                "borrowing_or_catalogue_text": False,
                "finished_tape_reversal": False,
                "symmetric_wrapper": False,
                "punctuation_changed_letter_tape": False,
            },
            "live_residual": {
                "owner": "left insertion holds the parent noel obligation until the right insertion discharges it",
                "residual": residual,
                "left_emission": LEFT_INSERTION,
                "right_emission": RIGHT_INSERTION,
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
                "effect": "exact growth evidence only; no readability certification",
            },
        },
        "novelty_preflight": {
            "status": "passed_before_candidate_generation",
            "snapshot_commit": "cbc523dd",
            "checked_with_ignore_rules_disabled": ["runs/", "experiments/", "docs/", "data/"],
            "queries": ["Noel solos", "Leon solos", "noelsolos", "leonsolos"],
            "prior_exact_clause_hits": 0,
            "prior_asymmetric_noel_residual_geometry": 0,
            "distinct_from": "the previous sees variant, rejected because its event stems were already present in generated runs",
            "previous_clause_duplicate_preserved_in": "runs/incumbent-568-asymmetric-residual-insertion-20260922.json",
        },
        "preserved_frontier": [
            {"artifact": str(PARENT.relative_to(ROOT)), "id": PARENT_ID, "letters": 568, "sha256": PARENT_SHA256},
            {"artifact": "runs/incumbent-550-central-event-bridge-20261002.json", "id": "central-distinct-events-560", "letters": 560, "sha256": "b5f98bfb0b44b31d8cbf78727672a74b588980e1fc8f1ff522a2c4ad1d800ccc"},
            {"artifact": "runs/incumbent-550-typed-center-product-20261002.json", "id": "typed-center-25", "letters": 558, "sha256": "29470b5ab408c402e8796530123357fea6a74aa4bdf14f7f1b2a601dbecc94fa"},
            {"artifact": "runs/incumbent-498-event-frame-seam-repair-20261002.json", "id": "depth39-longest-f1g1h1r", "letters": 556, "sha256": "28b303081c7eeae9b0f4c7e274d71e73551c64f5ad389b2d992b6183597f6d14"},
        ],
        "stats": {
            "exact_novel_candidate_count": 1,
            "longest_letters": len(tape),
            "growth_over_568": len(tape) - 568,
            "committed_character_contradictions": 0,
            "backtracks": 0,
        },
        "failed_probe_and_repair": {
            "failed_clause_trial": "Noel sees / Leon sees closed exactly at 584 letters but failed novelty preflight",
            "concrete_repair": "replace reused sees with new palindrome predicate solos; residual equation remains closed",
            "failed_central_seam": "map|s / s|pam bisects a word, so clause insertion there is invalid",
        },
        "next_repair": {
            "target": "the repeated inherited 'A tub? He maps' shell",
            "operator": "reopen its actual mirrored boundary and replace one repeated event frame with distinct event content under an explicit residual equation",
            "gate": "preserve 586 and the 568/560/558/556 frontier; keep exactness, novelty, and human readability as separate claims",
        },
    }


def main() -> None:
    payload = build_payload()
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
    print(payload["candidate"]["rendered"])


if __name__ == "__main__":
    main()
