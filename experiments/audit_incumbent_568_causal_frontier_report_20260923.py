#!/usr/bin/env python3
"""Verify the parent seam and preserve a report-only causal-chart obstruction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PARENT_PATH = ROOT / "runs" / "incumbent-560-outer-causal-scene-20261002.json"
OUT_PATH = ROOT / "runs" / "audit-incumbent-568-causal-frontier-report-20260923.json"
PARENT_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
LEFT_SPAN = (163, 194)
RIGHT_SPAN = (374, 405)
LEFT_FRONTIER = "After Mara rescued Mara, Mara rescued water."


def normalize(text: str) -> str:
    return "".join(char.lower() for char in text if char.isascii() and char.isalpha())


def outside_in(tape: str) -> dict[str, Any]:
    left, right = 0, len(tape) - 1
    while left < right:
        if tape[left] != tape[right]:
            return {"exact": False, "first_mismatch": [left, tape[left], right, tape[right]]}
        left += 1
        right -= 1
    return {"exact": True, "first_mismatch": None}


def build_payload() -> dict[str, Any]:
    source = json.loads(PARENT_PATH.read_text())
    row = next(item for item in source["rows"] if item["working_status"] == "working_length_incumbent")
    parent = str(row["rendered"])
    tape = normalize(parent)
    parent_sha = hashlib.sha256(tape.encode("ascii")).hexdigest()
    if len(tape) != 568 or parent_sha != PARENT_SHA256:
        raise AssertionError("pinned parent changed")
    left = tape[LEFT_SPAN[0] : LEFT_SPAN[1]]
    right = tape[RIGHT_SPAN[0] : RIGHT_SPAN[1]]
    if len(left) != 31 or left != right[::-1]:
        raise AssertionError("reported causal-chart seam does not match the pinned parent")
    frontier = normalize(LEFT_FRONTIER)
    residual = frontier[1:]
    if not frontier.startswith("a") or residual != "ftermararescuedmaramararescuedwater":
        raise AssertionError("reported one-character frontier residual changed")

    return {
        "experiment_id": "audit-incumbent-568-causal-frontier-report-20260923",
        "working_status": "report_only_zero_closure_not_rerun",
        "parent": {
            "artifact": str(PARENT_PATH.relative_to(ROOT)),
            "letters": len(tape),
            "sha256_normalized": parent_sha,
        },
        "operator": {
            "name": "connected two-event temporal/causal template chart",
            "provenance": "reported by Luna typed-chart lane; source seam and exposed frontier independently checked, search counters not rerun",
            "normalized_spans": [list(LEFT_SPAN), list(RIGHT_SPAN)],
            "source_tapes": [left, right],
            "source_letters_each": len(left),
        },
        "reported_search": {
            "fresh_connected_units": 254464,
            "exact_closures": 0,
            "furthest_prefix_letters": 1,
            "reported_frontier": LEFT_FRONTIER,
            "reported_left_cursor": 164,
            "reported_reverse_right_cursor": 403,
            "verified_left_frontier_tape": frontier,
            "verified_left_residual_after_prefix": residual,
            "opposing_state_status": "reported_empty after first character; not independently rerun",
        },
        "candidate": {
            "emitted": False,
            "rendered": None,
            "letters": None,
            "normalized_sha256": None,
            "exact_audit": None,
        },
        "admission": {
            "admitted": False,
            "reason": "No exact child or closed parse was reported. The connected event chart's best verified sample leaves `ftermararescuedmaramararescuedwater` after one opening `a`, while its opposing state set is reported empty. The chart search totals are testimony, not reproduced measurements.",
            "human_readability_certified": False,
            "reader_evidence": False,
        },
        "next_construction": "Retire this two-event chart and this geometry. Change to a different sentence-bounded seam and co-design a connected scene under the first live reverse-character obligation, with a strict growth target and flank grammar preflight.",
    }


if __name__ == "__main__":
    payload = build_payload()
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "status": payload["working_status"],
        "parent_letters": payload["parent"]["letters"],
        "source_letters": payload["operator"]["source_letters_each"],
        "reported_units_not_rerun": payload["reported_search"]["fresh_connected_units"],
        "verified_frontier_residual": payload["reported_search"]["verified_left_residual_after_prefix"],
        "candidate_emitted": payload["candidate"]["emitted"],
    }, indent=2))
