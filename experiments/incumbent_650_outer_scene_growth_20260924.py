"""Grow the 650-letter exact lineage with a fresh outer event pair."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.incumbent_498_deep_clause_transducer_20261002 import audit


PARENT = ROOT / "runs" / "incumbent-608-repeated-shell-repair-20261002.json"
OUT = ROOT / "runs" / "incumbent-650-outer-scene-growth-20260924.json"
PARENT_ID = "double-event-shell-repair-650"
PARENT_SHA256 = "2bd92686cbd01945be3869fbeee7ae9415cfcdae5cd616b26d89ec4f0acc54a9"
EXPECTED_SHA256 = "a190df41487ba2d1e1b55054690e4002c63f3a94123202b51aa4e502772b8acf"

OLD_LEFT = "A ram saw Nadia."
OLD_RIGHT = "Aidan was Mara."
NEW_LEFT = "Nadia stops a ram."
NEW_RIGHT = "Mara spots Aidan."


def normalize(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def build_payload() -> dict[str, object]:
    parent_payload = json.loads(PARENT.read_text(encoding="utf-8"))
    parent = next(row for row in parent_payload["rows"] if row["id"] == PARENT_ID)
    assert parent["audit"]["letters"] == 650
    assert parent["audit"]["sha256_forward"] == PARENT_SHA256
    assert parent["audit"]["two_pointer_exact"]
    assert parent["audit"]["project_validator_exact"]

    rendered = str(parent["rendered"])
    assert rendered.count(OLD_LEFT) == 1
    assert rendered.count(OLD_RIGHT) == 1

    left_tape = normalize(NEW_LEFT)
    right_tape = normalize(NEW_RIGHT)
    assert len(left_tape) == len(right_tape) == 14
    assert left_tape == right_tape[::-1]
    child = rendered.replace(OLD_LEFT, NEW_LEFT, 1).replace(OLD_RIGHT, NEW_RIGHT, 1)
    child_audit = audit(child)
    assert child_audit["letters"] == 654
    assert child_audit["sha256_forward"] == EXPECTED_SHA256
    assert child_audit["two_pointer_exact"]
    assert child_audit["byte_pointer_exact"]
    assert child_audit["project_validator_exact"]

    return {
        "experiment_id": "incumbent-650-outer-scene-growth-20260924",
        "method": "replace the exact outer saw/was shell with a fresh stop/spot event pair under live character ownership",
        "parent": {
            "artifact": str(PARENT.relative_to(ROOT)),
            "id": PARENT_ID,
            "letters": 650,
            "sha256": PARENT_SHA256,
        },
        "stats": {
            "independently_exact_children": 1,
            "longest_letters": 654,
            "growth_over_parent": 4,
            "committed_character_contradictions": 0,
        },
        "working_length_incumbent": {
            "artifact": str(OUT.relative_to(ROOT)),
            "id": "outer-stop-spot-event-654",
            "letters": 654,
            "sha256": EXPECTED_SHA256,
        },
        "preserved_lineage": [
            {
                "artifact": "runs/incumbent-560-outer-causal-scene-20261002.json",
                "id": "outer-causal-scene-568-working-incumbent",
                "letters": 568,
                "sha256": "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380",
            },
            {
                "artifact": "runs/incumbent-568-live-partial-seam-growth-20261002.json",
                "id": "live-nadia-seam-shell-repair-608",
                "letters": 608,
                "sha256": "eacd84ecc82aacb84fe19b557faf91494370e86c986248be5e37de9ad7298248",
            },
            {
                "artifact": str(PARENT.relative_to(ROOT)),
                "id": PARENT_ID,
                "letters": 650,
                "sha256": PARENT_SHA256,
            },
        ],
        "rows": [{
            "id": "outer-stop-spot-event-654",
            "working_status": "working_length_incumbent",
            "rendered": child,
            "audit": child_audit,
            "parent_artifact": str(PARENT.relative_to(ROOT)),
            "parent_id": PARENT_ID,
            "parent_sha256": PARENT_SHA256,
            "growth_over_parent": 4,
            "replacement": {
                "old_left": OLD_LEFT,
                "old_right": OLD_RIGHT,
                "new_left": NEW_LEFT,
                "new_right": NEW_RIGHT,
                "left_tape": left_tape,
                "right_tape": right_tape,
                "equation_exact": True,
                "new_event_content": ["Nadia stops a ram", "Mara spots Aidan"],
            },
            "live_seam": {
                "initial_owner": "left",
                "left_emission": left_tape,
                "right_obligation": left_tape[::-1],
                "right_consumption": right_tape,
                "final_residual": "",
                "backtracks": 0,
            },
            "repair_debt": {
                "inherits_rough_prose_and_repeated_scaffolding": True,
                "human_certified": False,
                "effect": "retain as a length-track child; do not present as reader-validated prose",
            },
            "next_reader_facing_test": {
                "status": "not_administered",
                "comparison": "blind randomized comparison of this child, its 650-letter parent, the 38-letter clean benchmark, intact English controls, and shuffled controls",
                "readability_measure": "human preference and connected-English ratings; no programmatic score certifies readability",
            },
            "next_operator": "reopen the remaining central repeated delivers/maps and spam's reviled shells with a fresh event equation and live residual ownership",
        }],
    }


def main() -> None:
    payload = build_payload()
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["stats"], sort_keys=True))
    print(payload["rows"][0]["rendered"])


if __name__ == "__main__":
    main()
