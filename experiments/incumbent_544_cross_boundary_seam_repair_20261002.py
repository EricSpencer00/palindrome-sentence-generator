"""Repair the 544 child's worst seam with a boundary-shifting event span."""
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


PARENT = ROOT / "runs" / "incumbent-534-inner-event-repair-20261002.json"
OUT = ROOT / "runs" / "incumbent-544-cross-boundary-seam-repair-20261002.json"
PARENT_SHA256 = "2ea2411e5fea4d27d3db24ba0e471cc6a52a4a2196b52fb6a82b5cc6033b0655"
EXPECTED_SHA256 = "3040f0c4ac28002aa0edd7ce2fd920751b10e4a4f5430d82de3e46d09b3e7673"

OLD_LEFT = "Noel, did I draw Mara? Was I God?"
OLD_RIGHT = "Dog I saw, Aram. Ward I did, Leon."
NEW_LEFT = "Noel, I sit. Pat notes. Mara saw God."
NEW_RIGHT = "Dog was Aram. Seton, tap. 'Tis I, Leon."


def load_parent() -> dict[str, object]:
    payload = json.loads(PARENT.read_text())
    row = next(item for item in payload["rows"] if item["id"] == "combined-finite-role-544")
    assert row["audit"]["letters"] == 544
    assert row["audit"]["sha256_forward"] == PARENT_SHA256
    assert audit(row["rendered"])["sha256_forward"] == PARENT_SHA256
    return row


def build_payload() -> dict[str, object]:
    parent = load_parent()
    parent_rendered = str(parent["rendered"])
    assert parent_rendered.count(OLD_LEFT) == 1
    assert parent_rendered.count(OLD_RIGHT) == 1

    old_left_tape = independent_tape(OLD_LEFT)
    old_right_tape = independent_tape(OLD_RIGHT)
    new_left_tape = independent_tape(NEW_LEFT)
    new_right_tape = independent_tape(NEW_RIGHT)
    assert old_left_tape[::-1] == old_right_tape
    assert new_left_tape[::-1] == new_right_tape

    rendered = parent_rendered.replace(OLD_LEFT, NEW_LEFT).replace(OLD_RIGHT, NEW_RIGHT)
    result_audit = audit(rendered)
    assert result_audit["letters"] == 550
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

    # Boundaries do not align under reflection: left ``Pat|notes`` becomes
    # right ``Seton|tap``.  Matching notes first consumes reverse(on) == no,
    # carries residual ``tes``, and only the following ``Set`` closes it.
    assert independent_tape("Pat notes")[::-1] == independent_tape("Seton tap")

    row = {
        "id": "cross-boundary-finite-events-550",
        "rendered": rendered,
        "audit": result_audit,
        "parent_artifact": str(PARENT.relative_to(ROOT)),
        "parent_sha256": PARENT_SHA256,
        "growth_over_parent": 6,
        "source_498_artifact": "runs/overhang-growth-from-240-20261001.json",
        "source_498_sha256": "e809a2a05a414347615f68f27f6b4974aa00ede287fdb2a4b23f6ffbadec9032",
        "replacement": {
            "old_left": OLD_LEFT,
            "old_right": OLD_RIGHT,
            "old_letters_per_side": len(old_left_tape),
            "new_left": NEW_LEFT,
            "new_right": NEW_RIGHT,
            "new_letters_per_side": len(new_left_tape),
            "reverse_exact": True,
            "new_finite_events": [
                "Noel sits",
                "Pat takes notes",
                "Mara sees God",
                "Dog is played by Aram",
                "Seton is told to tap",
                "Leon identifies himself",
            ],
        },
        "live_state": {
            "outer_owner": "R",
            "outer_residual_before_center": "won",
            "inner_owner_after_notes_vs_on": "R",
            "inner_residual": "tes",
            "inner_closure": "Set",
            "right_boundary_consumption": "Leon|won",
            "residual_after_shell": "",
        },
        "structural_audit": {
            "left_sentence_count": 3,
            "right_sentence_count": 3,
            "word_boundaries_reflect_one_to_one": False,
            "left_boundary": "Pat|notes",
            "right_boundary": "Seton|tap",
            "outer_wrapper_added": False,
            "fixed_finished_tape_reversal": False,
            "posthoc_character_repair": False,
            "human_certified": False,
        },
        "worst_remaining_seam": {
            "text": "I saw diaper. / Repaid was I",
            "diagnosis": "the central event is exact but semantically unmotivated and inverted",
            "next_operator": (
                "replace the central span with an original finite event while "
                "keeping a nonempty residual across its sentence boundary"
            ),
        },
    }
    return {
        "experiment_id": "incumbent-544-cross-boundary-seam-repair-20261002",
        "method": (
            "replace the predeclared worst interior seam with three finite "
            "events per side whose word boundaries shift under reflection"
        ),
        "parent": {
            "artifact": str(PARENT.relative_to(ROOT)),
            "letters": 544,
            "sha256": PARENT_SHA256,
        },
        "stats": {
            "authored_paths": 1,
            "independently_exact_children": 1,
            "children_over_530": 1,
            "longest_letters": 550,
            "interior_seams_repaired": 1,
        },
        "rows": [row],
    }


def main() -> None:
    payload = build_payload()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
    print(payload["rows"][0]["rendered"])


if __name__ == "__main__":
    main()
