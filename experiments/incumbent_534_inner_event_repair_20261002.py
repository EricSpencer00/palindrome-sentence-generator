"""Repair finite-event spans inside the 534-letter deep-seam child.

The prior experiment removed 258 inherited letters but left several malformed
clauses in its 240-letter center.  This experiment changes only named interior
spans.  Each replacement is authored as two readable finite clauses whose
normalized tapes are exact reverses; the live outer ``won`` residual remains
open until the unchanged ``Leon|won`` boundary.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.incumbent_498_deep_clause_transducer_20261002 import (
    CENTER_RENDERED,
    EXPECTED_SHA256 as PARENT_SHA256,
    LEFT_SHELL,
    RIGHT_SHELL,
    audit,
    independent_tape,
)


PARENT = ROOT / "runs" / "incumbent-498-deep-clause-transducer-20261002.json"
OUT = ROOT / "runs" / "incumbent-534-inner-event-repair-20261002.json"


@dataclass(frozen=True)
class Repair:
    repair_id: str
    old_left: str
    old_right: str
    new_left: str
    new_right: str
    left_offset: int
    right_offset: int
    kind: str
    new_event: str


ROLE_EVENT = Repair(
    repair_id="role-event",
    old_left="Nora, was I evil?",
    old_right="Live I saw, Aron.",
    new_left="Nora saw Noel live.",
    new_right="“Evil Leon” was Aron.",
    left_offset=12,
    right_offset=216,
    kind="finite performance event plus named role",
    new_event="Nora sees Noel perform; Aron plays Evil Leon",
)

FINITE_DELIVERY = Repair(
    repair_id="finite-delivery",
    old_left="Nora, I saw deliver Noel.",
    old_right="Leon. Reviled was I, Aron.",
    new_left="Nadia delivers maps. Leon,",
    new_right="Noel; spam's reviled, Aidan.",
    left_offset=91,
    right_offset=130,
    kind="cross-sentence finite-clause substitution",
    new_event="Nadia delivers maps before Leon reports what he saw",
)

SVO_EVENT = Repair(
    repair_id="svo-event",
    old_left="Sara, did I live?",
    old_right="Evil I did, Aras.",
    new_left="Sara spots rats.",
    new_right="Star stops Aras.",
    left_offset=47,
    right_offset=181,
    kind="subject-verb-object substitution",
    new_event="Sara spots rats; Star stops Aras",
)

SPECS = (
    ("single-svo-536", (SVO_EVENT,), "cb0ac4ab0c2a765c19e49b8760554aa28d0341e9d1d921df0e24bcb66990f1c8"),
    ("single-finite-delivery-538", (FINITE_DELIVERY,), "8bb6b80027e688df1d0ec9a56a5ed1be75adb661bcb572995efce70a9761f693"),
    ("single-role-event-540", (ROLE_EVENT,), "51cbdcce560bf88e7761c817ee5e15b498735c1a92786a175c12fd0e24113ca2"),
    (
        "combined-finite-role-544",
        (FINITE_DELIVERY, ROLE_EVENT),
        "2ea2411e5fea4d27d3db24ba0e471cc6a52a4a2196b52fb6a82b5cc6033b0655",
    ),
)


def apply_repair(center: str, repair: Repair) -> str:
    assert center.count(repair.old_left) == 1
    assert center.count(repair.old_right) == 1
    left_tape = independent_tape(repair.new_left)
    right_tape = independent_tape(repair.new_right)
    assert left_tape[::-1] == right_tape
    return center.replace(repair.old_left, repair.new_left).replace(
        repair.old_right, repair.new_right
    )


def repair_trace(repair: Repair) -> dict[str, object]:
    old_left_tape = independent_tape(repair.old_left)
    old_right_tape = independent_tape(repair.old_right)
    new_left_tape = independent_tape(repair.new_left)
    new_right_tape = independent_tape(repair.new_right)
    assert old_left_tape[::-1] == old_right_tape
    assert new_left_tape[::-1] == new_right_tape
    return {
        "id": repair.repair_id,
        "kind": repair.kind,
        "center_offsets_before": [
            [repair.left_offset, repair.left_offset + len(old_left_tape)],
            [repair.right_offset, repair.right_offset + len(old_right_tape)],
        ],
        "old_left": repair.old_left,
        "old_right": repair.old_right,
        "new_left": repair.new_left,
        "new_right": repair.new_right,
        "old_letters_per_side": len(old_left_tape),
        "new_letters_per_side": len(new_left_tape),
        "replacement_reverse_exact": True,
        "new_event_content": repair.new_event,
    }


def build_payload() -> dict[str, object]:
    parent_payload = json.loads(PARENT.read_text())
    parent_row = parent_payload["rows"][0]
    assert parent_row["audit"]["letters"] == 534
    assert parent_row["audit"]["sha256_forward"] == PARENT_SHA256

    rows = []
    for candidate_id, repairs, expected_sha in SPECS:
        center = CENTER_RENDERED
        for repair in repairs:
            center = apply_repair(center, repair)
        rendered = f"{LEFT_SHELL} {center} {RIGHT_SHELL}"
        result_audit = audit(rendered)
        assert result_audit["letters"] > 530
        assert result_audit["sha256_forward"] == expected_sha
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
        rows.append({
            "id": candidate_id,
            "rendered": rendered,
            "audit": result_audit,
            "parent_artifact": str(PARENT.relative_to(ROOT)),
            "parent_sha256": PARENT_SHA256,
            "growth_over_534": result_audit["letters"] - 534,
            "retained_498_parent_letters_before_repairs": 240,
            "repairs": [repair_trace(repair) for repair in repairs],
            "live_state": {
                "owner": "R",
                "residual_before_center": "won",
                "right_boundary_consumption": "Leon|won",
                "residual_after_shell": "",
            },
            "provenance": {
                "targeted_inner_repairs": True,
                "outer_wrapper_added": False,
                "larger_pair_bank_sweep": False,
                "finished_tape_reversal": False,
                "posthoc_character_repair": False,
                "human_certified": False,
            },
        })

    rows.sort(key=lambda row: (-row["audit"]["letters"], row["id"]))
    return {
        "experiment_id": "incumbent-534-inner-event-repair-20261002",
        "method": (
            "target exact interior clause spans in the 534-letter child, "
            "substituting finite events before rendering while preserving the "
            "live outer residual"
        ),
        "parent": {
            "artifact": str(PARENT.relative_to(ROOT)),
            "letters": 534,
            "sha256": PARENT_SHA256,
            "source_498_artifact": "runs/overhang-growth-from-240-20261001.json",
            "source_498_sha256": "e809a2a05a414347615f68f27f6b4974aa00ede287fdb2a4b23f6ffbadec9032",
        },
        "stats": {
            "authored_repair_paths": len(rows),
            "independently_exact_children": len(rows),
            "children_over_530": sum(row["audit"]["letters"] > 530 for row in rows),
            "shortest_letters": min(row["audit"]["letters"] for row in rows),
            "longest_letters": max(row["audit"]["letters"] for row in rows),
        },
        "active_frontier": [
            "combined-finite-role-544",
            "single-role-event-540",
            "single-finite-delivery-538",
            "single-svo-536",
        ],
        "worst_remaining_seam": {
            "text": "Noel, did I draw Mara? / Ward I did, Leon.",
            "center_offsets": [[24, 40], [200, 216]],
            "next_operator": (
                "replace the paired interrogative/inversion with a new finite "
                "event whose word boundaries cross the reflected span"
            ),
        },
        "rows": rows,
    }


def main() -> None:
    payload = build_payload()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
    print(payload["rows"][0]["rendered"])


if __name__ == "__main__":
    main()
