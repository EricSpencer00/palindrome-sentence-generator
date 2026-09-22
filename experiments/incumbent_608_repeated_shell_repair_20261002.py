"""Repair two repeated delivers/reviled shells in the 608-letter incumbent."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.incumbent_498_deep_clause_transducer_20261002 import audit
from experiments.incumbent_550_wide_center_story_20261002 import proper_spans


PARENT = ROOT / "runs" / "incumbent-568-live-partial-seam-growth-20261002.json"
OUT = ROOT / "runs" / "incumbent-608-repeated-shell-repair-20261002.json"
PARENT_ID = "live-nadia-seam-shell-repair-608"
PARENT_SHA256 = "eacd84ecc82aacb84fe19b557faf91494370e86c986248be5e37de9ad7298248"
EXPECTED_SHA256 = "2bd92686cbd01945be3869fbeee7ae9415cfcdae5cd616b26d89ec4f0acc54a9"

REPAIRS = [
    {
        "old_left": "Nora delivers maps.",
        "old_right": "Spam's reviled, Aron.",
        "new_left": "Nora spots a rat. Nadia spots a ram.",
        "new_right": "Mara stops Aidan. Tara stops Aron.",
    },
    {
        "old_left": "Aidan delivers maps.",
        "old_right": "Spam's reviled, Nadia.",
        "new_left": "Aidan spots a rat. Nora spots a ram.",
        "new_right": "Mara stops Aron. Tara stops Nadia.",
    },
]


def normalize(text: str) -> str:
    return "".join(re.findall(r"[a-z]", text.lower()))


def build_payload() -> dict[str, object]:
    parent_payload = json.loads(PARENT.read_text())
    parent = next(row for row in parent_payload["rows"] if row["id"] == PARENT_ID)
    assert parent["audit"]["letters"] == 608
    assert parent["audit"]["sha256_forward"] == PARENT_SHA256
    rendered = str(parent["rendered"])

    equations: list[dict[str, object]] = []
    for repair in REPAIRS:
        old_left = repair["old_left"]
        old_right = repair["old_right"]
        new_left = repair["new_left"]
        new_right = repair["new_right"]
        assert rendered.count(old_left) == 1
        assert rendered.count(old_right) == 1
        assert normalize(new_left) == normalize(new_right)[::-1]
        rendered = rendered.replace(old_left, new_left, 1)
        rendered = rendered.replace(old_right, new_right, 1)
        equations.append({
            **repair,
            "left_tape": normalize(new_left),
            "right_tape": normalize(new_right),
            "equation_exact": True,
            "final_owner": None,
            "final_residual": "",
            "backtracks": 0,
        })

    result_audit = audit(rendered)
    assert result_audit["letters"] == 650
    assert result_audit["sha256_forward"] == EXPECTED_SHA256
    assert result_audit["two_pointer_exact"]
    assert result_audit["byte_pointer_exact"]
    assert result_audit["project_validator_exact"]

    repetition_delta = {
        phrase: {
            "before": str(parent["rendered"]).lower().count(phrase),
            "after": rendered.lower().count(phrase),
        }
        for phrase in ["delivers maps", "spam's reviled"]
    }
    assert all(delta == {"before": 4, "after": 2} for delta in repetition_delta.values())
    spans = proper_spans(rendered)

    row = {
        "id": "double-event-shell-repair-650",
        "working_status": "working_length_incumbent",
        "rendered": rendered,
        "audit": result_audit,
        "parent_artifact": str(PARENT.relative_to(ROOT)),
        "parent_id": PARENT_ID,
        "parent_sha256": PARENT_SHA256,
        "growth_over_parent": 42,
        "new_event_content": [
            "Nora spots a rat",
            "Nadia spots a ram",
            "Aidan spots a rat",
            "Nora spots a ram",
            "Mara stops Aidan",
            "Tara stops Aron",
            "Mara stops Aron",
            "Tara stops Nadia",
        ],
        "live_repairs": equations,
        "repetition_delta": repetition_delta,
        "repair_debt": {
            "proper_palindromic_spans": len(spans),
            "rough_prose": True,
            "human_certified": False,
            "effect": "continue event diversification; do not reject exact growth",
        },
        "provenance": "two deterministic authored mirrored-shell equations over the verified 608 tape",
        "next_operator": (
            "retain the 650-letter tape; reopen the remaining central "
            "delivers/reviled shell and replace it with a nonrepeated event "
            "equation while preserving exactness"
        ),
    }

    return {
        "experiment_id": "incumbent-608-repeated-shell-repair-20261002",
        "method": "paired event-shell substitution with live exact equations",
        "parent": {
            "artifact": str(PARENT.relative_to(ROOT)),
            "id": PARENT_ID,
            "letters": 608,
            "sha256": PARENT_SHA256,
        },
        "stats": {
            "independently_exact_children": 1,
            "longest_letters": 650,
            "growth_letters": 42,
            "shells_repaired": 2,
            "committed_character_contradictions": 0,
            "backtracks": 0,
        },
        "working_length_incumbent": {
            "artifact": str(OUT.relative_to(ROOT)),
            "id": row["id"],
            "letters": 650,
            "sha256": EXPECTED_SHA256,
        },
        "preserved_frontier": [
            {
                "artifact": str(PARENT.relative_to(ROOT)),
                "id": PARENT_ID,
                "letters": 608,
                "sha256": PARENT_SHA256,
            },
            {
                "artifact": "runs/incumbent-560-outer-causal-scene-20261002.json",
                "id": "outer-causal-scene-568-working-incumbent",
                "letters": 568,
                "sha256": "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380",
            },
            {
                "artifact": "runs/incumbent-550-central-event-bridge-20261002.json",
                "id": "central-distinct-events-560",
                "letters": 560,
                "sha256": "b5f98bfb0b44b31d8cbf78727672a74b588980e1fc8f1ff522a2c4ad1d800ccc",
            },
            {
                "artifact": "runs/incumbent-498-event-frame-seam-repair-20261002.json",
                "id": "depth39-longest-f1g1h1r",
                "letters": 556,
                "sha256": "28b303081c7eeae9b0f4c7e274d71e73551c64f5ad389b2d992b6183597f6d14",
            },
        ],
        "rows": [row],
    }


def main() -> None:
    payload = build_payload()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
    print(payload["rows"][0]["rendered"])


if __name__ == "__main__":
    main()
