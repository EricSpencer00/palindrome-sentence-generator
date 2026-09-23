"""Repair the 586-letter outer frame with a causally linked event pair."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.validator import is_palindrome, normalize


PARENT = ROOT / "runs" / "incumbent-586-repeated-shell-valency-repair-20260922.json"
OUT = ROOT / "runs" / "incumbent-586-outer-flow-frame-repair-20260923.json"
PARENT_ID = "asymmetric-noel-leon-solos-586-one-shell-repaired"
PARENT_SHA256 = "142f34541db73cb8a7b9c7fa96cb34abec9fd4bea5d3003b6482d476f6e7569b"
OLD_LEFT = "Wolf spots Nora."
NEW_LEFT = "Wolf spots flow."
OLD_RIGHT = "Aron stops flow now, Noel."
NEW_RIGHT = "Wolf stops flow now, Noel."
LEFT_START = 7
RIGHT_START = 566
PAIR_LETTERS = 13


def tape(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def two_pointer_exact(value: str) -> bool:
    left, right = 0, len(value) - 1
    while left < right:
        if value[left] != value[right]:
            return False
        left += 1
        right -= 1
    return bool(value)


def residual_trace(left: str, right: str) -> list[dict[str, object]]:
    assert len(left) == len(right) == PAIR_LETTERS
    residual = left
    trace: list[dict[str, object]] = []
    for cursor, emitted in enumerate(left):
        expected = right[-1 - cursor]
        assert emitted == expected
        residual = residual[1:]
        trace.append(
            {
                "owner": "left-clause-emitter",
                "cursor": cursor,
                "emitted": emitted,
                "right_cursor": len(right) - 1 - cursor,
                "expected": expected,
                "residual_after": residual,
            }
        )
    assert residual == ""
    return trace


def build_payload() -> dict[str, object]:
    parent_payload = json.loads(PARENT.read_text())
    parent_row = parent_payload["candidate"]
    assert parent_row["id"] == PARENT_ID
    parent_rendered = str(parent_row["rendered"])
    parent_tape = tape(parent_rendered)
    assert len(parent_tape) == 586 and two_pointer_exact(parent_tape)
    assert hashlib.sha256(parent_tape.encode("ascii")).hexdigest() == PARENT_SHA256
    assert parent_tape[LEFT_START : LEFT_START + PAIR_LETTERS] == tape(OLD_LEFT)
    assert parent_tape[RIGHT_START : RIGHT_START + PAIR_LETTERS] == tape("Aron stops flow")
    assert parent_rendered.count(OLD_LEFT) == 1
    assert parent_rendered.count(OLD_RIGHT) == 1

    rendered = parent_rendered.replace(OLD_LEFT, NEW_LEFT, 1)
    rendered = rendered.replace(OLD_RIGHT, NEW_RIGHT, 1)
    candidate_tape = tape(rendered)
    forward_sha = hashlib.sha256(candidate_tape.encode("ascii")).hexdigest()
    reverse_sha = hashlib.sha256(candidate_tape[::-1].encode("ascii")).hexdigest()
    left_tape = tape(NEW_LEFT)
    right_tape = tape("Wolf stops flow")
    trace = residual_trace(left_tape, right_tape)

    assert len(candidate_tape) == len(parent_tape) == 586
    assert candidate_tape[LEFT_START : LEFT_START + PAIR_LETTERS] == left_tape
    assert candidate_tape[RIGHT_START : RIGHT_START + PAIR_LETTERS] == right_tape
    assert left_tape[::-1] == right_tape
    assert two_pointer_exact(candidate_tape)
    assert is_palindrome(rendered)
    assert normalize(rendered) == candidate_tape
    assert forward_sha == reverse_sha
    assert rendered.startswith("Leon won. Wolf spots flow.")
    assert rendered.endswith("Wolf stops flow now, Noel.")

    return {
        "experiment_id": "incumbent-586-outer-flow-frame-repair-20260923",
        "method": "single outer-frame semantic repair using a common-noun role bridge",
        "status": "exact_same_length_outer_frame_repair_with_inherited_prose_debt",
        "parent": {
            "artifact": str(PARENT.relative_to(ROOT)),
            "id": PARENT_ID,
            "letters": len(parent_tape),
            "sha256": PARENT_SHA256,
            "edited_normalized_spans": {
                "left": [LEFT_START, LEFT_START + PAIR_LETTERS],
                "right": [RIGHT_START, RIGHT_START + PAIR_LETTERS],
            },
        },
        "candidate": {
            "id": "outer-wolf-flow-scene-586",
            "rendered": rendered,
            "letters": len(candidate_tape),
            "sha256": forward_sha,
            "audit": {
                "independent_two_pointer_exact": two_pointer_exact(candidate_tape),
                "project_validator_exact": bool(is_palindrome(rendered)),
                "project_normalizer_agrees": normalize(rendered) == candidate_tape,
                "sha256_forward": forward_sha,
                "sha256_reverse": reverse_sha,
                "sha_equal": forward_sha == reverse_sha,
            },
            "live_seam": {
                "parent_left": tape(OLD_LEFT),
                "parent_right": tape("Aron stops flow"),
                "left_emission": left_tape,
                "right_obligation": right_tape,
                "equation": "wolfspotsflow = reverse(wolfstopsflow)",
                "initial_owner": "left-clause-emitter",
                "trace": trace,
                "final_owner": None,
                "final_residual": "",
                "committed_character_contradictions": 0,
            },
            "repair": {
                "before_left": OLD_LEFT,
                "after_left": NEW_LEFT,
                "before_reflected_support": "Aron stops flow",
                "after_reflected_support": "Wolf stops flow",
                "rendered_outer_frame": [NEW_LEFT, NEW_RIGHT],
                "interpretation": "Wolf notices the flow, then stops it; the final vocative addresses Noel.",
            },
            "provenance": {
                "source_is_generated_parent": True,
                "borrowed_or_catalogue_text": False,
                "punctuation_changed_letter_tape": False,
                "finished_tape_reversal": False,
                "new_clause_collision_count_in_runs_experiments_docs_data": 0,
            },
            "repair_debt": {
                "remaining_A_tub_shell": True,
                "inherited_proper_palindromic_spans": True,
                "other_rough_prose": True,
                "human_readability_certified": False,
                "effect": "preserve as an outer-frame repair variant, not a readability claim",
            },
        },
        "novelty_preflight": {
            "status": "passed_before_this_candidate",
            "checked_with_ignore_rules_disabled": ["runs/", "experiments/", "docs/", "data/"],
            "queries": ["Wolf spots flow", "Wolf stops flow", "wolfspotsflow", "wolfstopsflow"],
            "prior_exact_clause_collisions": 0,
            "operator_overlap": "not a clause-bank or event-lattice sweep; one outer mirrored support is edited to make the same agent-theme pair form a linked notice-then-stop frame",
        },
        "stats": {
            "exact_children": 1,
            "letters": len(candidate_tape),
            "length_delta": 0,
            "outer_clause_pairs_repaired": 1,
            "backtracks": 0,
        },
        "next_construction": {
            "target": "the remaining A tub? He maps Nora seam in this 586-letter parent",
            "operator": "expand beyond the 13-letter question shell and jointly rederive adjacent clause boundaries while carrying typed dialogue state and character residual",
            "avoid": [
                "fixed 13-letter window",
                "complete-SVO event bank sweeps",
                "previous discourse-linked reverse-chain operator",
            ],
            "reader_test": "after a reader-worthy candidate emerges, randomized blinded evaluation against intact-prose and shuffled controls; programmatic measures remain diagnostic",
        },
    }


def main() -> None:
    payload = build_payload()
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
    print(payload["candidate"]["sha256"])
    print(payload["candidate"]["rendered"])


if __name__ == "__main__":
    main()
