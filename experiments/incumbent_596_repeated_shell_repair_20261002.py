"""Repair one inherited repeated shell in a direct 568-lineage 596 child."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.incumbent_498_deep_clause_transducer_20261002 import audit


PARENT = ROOT / "runs" / "incumbent-568-won-now-seam-growth-20261002.json"
OUT = ROOT / "runs" / "incumbent-596-repeated-shell-repair-20261002.json"
PARENT_ID = "won-now-nadia-596"
PARENT_SHA256 = "c48b3e11531d2a81b4f7a1c525556fd22d3098e949455de6a6be83ab00aadc3a"
EXPECTED_SHA256 = "06fcb154a3e1863d0caf0947a111716835def706d0aa87b8e7048893374bb923"

OLD_LEFT = "Mara stops rats. A tub? He maps Aron."
OLD_RIGHT = "Nora, spam. Eh, but a star spots Aram."
NEW_LEFT = "Mara spots a rat. Nora spots a ram."
NEW_RIGHT = "Mara stops Aron. Tara stops Aram."


def normalize(text: str) -> str:
    return "".join(re.findall(r"[a-z]", text.lower()))


def build_payload() -> dict[str, object]:
    parent_payload = json.loads(PARENT.read_text())
    parent = next(row for row in parent_payload["rows"] if row["id"] == PARENT_ID)
    assert parent["audit"]["letters"] == 596
    assert parent["audit"]["sha256_forward"] == PARENT_SHA256
    parent_rendered = str(parent["rendered"])
    assert parent_rendered.count(OLD_LEFT) == 1
    assert parent_rendered.count(OLD_RIGHT) == 1
    assert normalize(NEW_LEFT) == normalize(NEW_RIGHT)[::-1]

    rendered = parent_rendered.replace(OLD_LEFT, NEW_LEFT, 1)
    rendered = rendered.replace(OLD_RIGHT, NEW_RIGHT, 1)
    result_audit = audit(rendered)
    assert result_audit["letters"] == 594
    assert result_audit["sha256_forward"] == EXPECTED_SHA256
    assert result_audit["two_pointer_exact"]
    assert result_audit["byte_pointer_exact"]
    assert result_audit["project_validator_exact"]

    repeated_phrases = ["Mara stops rats.", "A tub?", "Eh, but a star spots Aram."]
    repetition_delta = {
        phrase: {
            "before": parent_rendered.count(phrase),
            "after": rendered.count(phrase),
        }
        for phrase in repeated_phrases
    }
    assert all(delta == {"before": 2, "after": 1} for delta in repetition_delta.values())

    row = {
        "id": "won-now-nadia-shell-repair-594",
        "working_status": "568_lineage_repair_frontier",
        "rendered": rendered,
        "audit": result_audit,
        "lineage_root_artifact": "runs/incumbent-560-outer-causal-scene-20261002.json",
        "lineage_root_sha256": "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380",
        "parent_artifact": str(PARENT.relative_to(ROOT)),
        "parent_id": PARENT_ID,
        "parent_sha256": PARENT_SHA256,
        "new_event_content": [
            "Mara spots a rat",
            "Nora spots a ram",
            "Mara stops Aron",
            "Tara stops Aram",
        ],
        "live_repair": {
            "old_left": OLD_LEFT,
            "old_right": OLD_RIGHT,
            "new_left": NEW_LEFT,
            "new_right": NEW_RIGHT,
            "left_emission": normalize(NEW_LEFT),
            "right_obligation": normalize(NEW_LEFT)[::-1],
            "right_consumption": normalize(NEW_RIGHT),
            "final_owner": None,
            "final_residual": "",
            "committed_character_contradictions": 0,
            "backtracks": 0,
        },
        "repetition_delta": repetition_delta,
        "repair_debt": {
            "inherited_proper_spans": True,
            "remaining_repeated_scaffolding": True,
            "rough_syntax": True,
            "human_certified": False,
            "effect": "continue repairing this exact 568-lineage frontier without demoting 568",
        },
        "provenance": "deterministic mirrored event substitution inside the direct 596 child",
    }

    return {
        "experiment_id": "incumbent-596-repeated-shell-repair-20261002",
        "method": "mirrored finite-event substitution inside a direct 568-lineage child",
        "working_incumbent": {
            "artifact": "runs/incumbent-560-outer-causal-scene-20261002.json",
            "id": "outer-causal-scene-568-working-incumbent",
            "letters": 568,
            "sha256": "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380",
        },
        "stats": {
            "independently_exact_children": 1,
            "repaired_child_letters": 594,
            "repeated_phrases_reduced": 3,
            "committed_character_contradictions": 0,
            "backtracks": 0,
        },
        "preserved_frontier_letters": [568, 560, 558, 556],
        "rows": [row],
        "next_operator": (
            "retain the 594 repaired child and the unrepaired 596 siblings; "
            "repair the next inherited repeated shell with a distinct event equation"
        ),
    }


def main() -> None:
    payload = build_payload()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
    print(payload["rows"][0]["rendered"])


if __name__ == "__main__":
    main()
