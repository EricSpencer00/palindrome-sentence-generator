"""Replace the 550 child's catalogue-like center with a finite event bridge."""
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.incumbent_498_deep_clause_transducer_20261002 import (
    audit,
    independent_tape,
)


PARENT = ROOT / "runs" / "incumbent-544-cross-boundary-seam-repair-20261002.json"
OUT = ROOT / "runs" / "incumbent-550-central-event-bridge-20261002.json"
PARENT_SHA256 = "3040f0c4ac28002aa0edd7ce2fd920751b10e4a4f5430d82de3e46d09b3e7673"
EXPECTED_SHA256 = "b5f98bfb0b44b31d8cbf78727672a74b588980e1fc8f1ff522a2c4ad1d800ccc"

OLD_WINDOW = "Leon, I saw diaper. Repaid was I, Noel;"
NEW_WINDOW = "Leon. Ari delivers maps. Spam's reviled, Ira. Noel;"
OLD_CENTER = "I saw diaper. Repaid was I"
NEW_LEFT = "Ari delivers maps."
NEW_RIGHT = "Spam's reviled, Ira."


def load_parent() -> dict[str, object]:
    payload = json.loads(PARENT.read_text())
    row = payload["rows"][0]
    assert row["audit"]["letters"] == 550
    assert row["audit"]["sha256_forward"] == PARENT_SHA256
    assert audit(row["rendered"])["sha256_forward"] == PARENT_SHA256
    return row


def build_payload() -> dict[str, object]:
    parent = load_parent()
    parent_rendered = str(parent["rendered"])
    assert parent_rendered.count(OLD_WINDOW) == 1

    old_center_tape = independent_tape(OLD_CENTER)
    new_left_tape = independent_tape(NEW_LEFT)
    new_right_tape = independent_tape(NEW_RIGHT)
    assert old_center_tape == old_center_tape[::-1]
    assert new_left_tape[::-1] == new_right_tape
    assert "ari" != "ira"

    rendered = parent_rendered.replace(OLD_WINDOW, NEW_WINDOW)
    result_audit = audit(rendered)
    assert result_audit["letters"] == 560
    assert result_audit["sha256_forward"] == EXPECTED_SHA256
    assert all(
        result_audit[key]
        for key in (
            "independent_normalizer_agrees",
            "two_pointer_exact",
            "byte_pointer_exact",
            "project_validator_exact",
            "sha_equal",
        )
    )

    return {
        "experiment_id": "incumbent-550-central-event-bridge-20261002",
        "method": (
            "replace the self-contained diaper/repaid midpoint with two "
            "distinct finite events whose word boundaries stagger across the center"
        ),
        "parent": {
            "artifact": str(PARENT.relative_to(ROOT)),
            "letters": 550,
            "sha256": PARENT_SHA256,
        },
        "stats": {
            "authored_paths": 1,
            "independently_exact_children": 1,
            "children_over_556_control": 1,
            "longest_letters": 560,
            "central_catalogue_spans_removed": 1,
        },
        "rows": [
            {
                "id": "central-distinct-events-560",
                "rendered": rendered,
                "audit": result_audit,
                "parent_artifact": str(PARENT.relative_to(ROOT)),
                "parent_sha256": PARENT_SHA256,
                "growth_over_parent": 10,
                "source_498_artifact": "runs/overhang-growth-from-240-20261001.json",
                "source_498_sha256": "e809a2a05a414347615f68f27f6b4974aa00ede287fdb2a4b23f6ffbadec9032",
                "replacement": {
                    "old_center": OLD_CENTER,
                    "old_center_letters": len(old_center_tape),
                    "new_left": NEW_LEFT,
                    "new_right": NEW_RIGHT,
                    "new_letters_per_side": len(new_left_tape),
                    "reverse_exact": True,
                    "new_lexical_content": ["Ari", "Ira"],
                    "new_event_content": [
                        "Ari delivers maps",
                        "Ira is told that spam is reviled",
                    ],
                },
                "live_state": {
                    "owner_after_reviled_vs_delivers": "R",
                    "residual": "s",
                    "residual_source": "the final s of spam's",
                    "closure": "maps consumes reverse(spam)",
                    "outer_owner": "R",
                    "outer_residual_before_center": "won",
                    "outer_boundary_consumption": "Leon|won",
                    "residual_after_shell": "",
                },
                "structural_audit": {
                    "left_sentence_count": 1,
                    "right_sentence_count": 1,
                    "names_distinct": True,
                    "self_palindromic_word_used": False,
                    "catalogue_text_used": False,
                    "word_boundaries_reflect_one_to_one": False,
                    "left_boundary": "deliver|s maps",
                    "right_boundary": "spam|s reviled",
                    "outer_wrapper_added": False,
                    "fixed_finished_tape_reversal": False,
                    "posthoc_character_repair": False,
                    "human_certified": False,
                },
                "worst_remaining_seam": {
                    "text": "the repeated outer map/rat delivery sequence",
                    "diagnosis": (
                        "locally finite but discourse-level repetition still "
                        "prevents a reader-worthy paragraph"
                    ),
                    "next_operator": (
                        "replace one complete outer event sequence with a "
                        "single causally linked scene transition while keeping "
                        "the existing live residual"
                    ),
                },
            }
        ],
    }


def main() -> None:
    payload = build_payload()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
    print(payload["rows"][0]["rendered"])


if __name__ == "__main__":
    main()
