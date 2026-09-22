"""Grow the verified 568 incumbent through its live w|on ... no|w seam."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.incumbent_498_deep_clause_transducer_20261002 import audit


PARENT = ROOT / "runs" / "incumbent-560-outer-causal-scene-20261002.json"
OUT = ROOT / "runs" / "incumbent-568-won-now-seam-growth-20261002.json"
PARENT_ID = "outer-causal-scene-568-working-incumbent"
PARENT_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"

SHELLS = [
    {
        "id": "won-now-nora-594",
        "left": "Nora spots a ram. Leon w",
        "right": "w, Noel. Mara stops Aron.",
        "events": ["Nora spots a ram", "Mara stops Aron"],
        "letters": 594,
        "sha256": "2d0ab9c00beeb4b8b221929d6d87c19bfc184ce235602b7e56905ff0f2754500",
    },
    {
        "id": "won-now-nadia-596",
        "left": "Nadia spots a ram. Leon w",
        "right": "w, Noel. Mara stops Aidan.",
        "events": ["Nadia spots a ram", "Mara stops Aidan"],
        "letters": 596,
        "sha256": "c48b3e11531d2a81b4f7a1c525556fd22d3098e949455de6a6be83ab00aadc3a",
    },
    {
        "id": "won-now-aidan-596",
        "left": "Aidan spots a rat. Leon w",
        "right": "w, Noel. Tara stops Nadia.",
        "events": ["Aidan spots a rat", "Tara stops Nadia"],
        "letters": 596,
        "sha256": "f742dac4368dc834452ad86a9adcddcfc67ec4ef4cb12a0e4be9e777cda66b99",
    },
]


def normalize(text: str) -> str:
    return "".join(re.findall(r"[a-z]", text.lower()))


def raw_boundary_after_letters(text: str, count: int) -> int:
    seen = 0
    for index, char in enumerate(text):
        if "a" <= char.lower() <= "z":
            seen += 1
            if seen == count:
                return index + 1
    raise ValueError(f"surface has fewer than {count} letters")


def build_payload() -> dict[str, object]:
    parent_payload = json.loads(PARENT.read_text())
    parent = parent_payload["rows"][0]
    assert parent["id"] == PARENT_ID
    assert parent["audit"]["letters"] == 568
    assert parent["audit"]["sha256_forward"] == PARENT_SHA256
    parent_rendered = str(parent["rendered"])
    parent_tape = normalize(parent_rendered)
    assert parent_tape == parent_tape[::-1]

    left_raw = raw_boundary_after_letters(parent_rendered, 5)
    right_raw = raw_boundary_after_letters(parent_rendered, len(parent_tape) - 5)
    retained_rendered = parent_rendered[left_raw:right_raw]
    retained_tape = normalize(retained_rendered)
    assert parent_rendered[:left_raw] == "Leon w"
    assert parent_rendered[right_raw:] == "w, Noel."
    assert retained_tape == parent_tape[5:-5]
    assert retained_tape == retained_tape[::-1]
    assert retained_rendered.startswith("on.")
    assert retained_rendered.endswith("no")

    rows: list[dict[str, object]] = []
    for shell in SHELLS:
        left = str(shell["left"])
        right = str(shell["right"])
        left_tape = normalize(left)
        right_tape = normalize(right)
        assert left_tape == right_tape[::-1]
        rendered = left + retained_rendered + right
        result_audit = audit(rendered)
        assert result_audit["letters"] == shell["letters"]
        assert result_audit["sha256_forward"] == shell["sha256"]
        assert result_audit["two_pointer_exact"]
        assert result_audit["byte_pointer_exact"]
        assert result_audit["project_validator_exact"]
        rows.append({
            "id": shell["id"],
            "working_status": "568_lineage_growth_frontier",
            "rendered": rendered,
            "audit": result_audit,
            "parent_artifact": str(PARENT.relative_to(ROOT)),
            "parent_id": PARENT_ID,
            "parent_sha256": PARENT_SHA256,
            "growth_over_parent": int(shell["letters"]) - 568,
            "new_event_content": shell["events"],
            "live_seam": {
                "parent_left_cursor": 5,
                "parent_right_cursor_exclusive": 563,
                "retained_letters": 558,
                "left_partial_join": "w|on",
                "right_partial_join": "no|w",
                "initial_owner": "left",
                "left_emission": left_tape,
                "right_obligation": left_tape[::-1],
                "right_consumption": right_tape,
                "final_owner": None,
                "final_residual": "",
                "committed_character_contradictions": 0,
                "backtracks": 0,
            },
            "provenance": "deterministic authored event shell over an actual partial-word seam of the verified 568 tape",
            "repair_debt": {
                "inherited_proper_spans": True,
                "inherited_repeated_scaffolding": True,
                "rough_syntax": True,
                "human_certified": False,
                "effect": "retain as exact growth frontier; do not demote the 568 incumbent",
            },
        })

    return {
        "experiment_id": "incumbent-568-won-now-seam-growth-20261002",
        "method": "live partial-word residual ownership at w|on ... no|w",
        "working_incumbent": {
            "artifact": str(PARENT.relative_to(ROOT)),
            "id": PARENT_ID,
            "letters": 568,
            "sha256": PARENT_SHA256,
        },
        "stats": {
            "independently_exact_children": len(rows),
            "children_longer_than_568": len(rows),
            "longest_letters": max(row["audit"]["letters"] for row in rows),
            "committed_character_contradictions": 0,
            "backtracks": 0,
        },
        "preserved_frontier": [
            {
                "artifact": str(PARENT.relative_to(ROOT)),
                "id": PARENT_ID,
                "letters": 568,
                "sha256": PARENT_SHA256,
            },
            {
                "artifact": "runs/incumbent-550-central-event-bridge-20261002.json",
                "id": "central-distinct-events-560",
                "letters": 560,
                "sha256": "b5f98bfb0b44b31d8cbf78727672a74b588980e1fc8f1ff522a2c4ad1d800ccc",
            },
            {
                "artifact": "runs/incumbent-550-typed-center-product-20261002.json",
                "id": "typed-center-25",
                "letters": 558,
                "sha256": "29470b5ab408c402e8796530123357fea6a74aa4bdf14f7f1b2a601dbecc94fa",
            },
            {
                "artifact": "runs/incumbent-498-event-frame-seam-repair-20261002.json",
                "id": "depth39-longest-f1g1h1r",
                "letters": 556,
                "sha256": "28b303081c7eeae9b0f4c7e274d71e73551c64f5ad389b2d992b6183597f6d14",
            },
        ],
        "rows": rows,
        "next_operator": (
            "keep 568 as the working incumbent and repair the clearest new "
            "596-letter child at an inherited repeated shell; change seams "
            "only on a committed normalized-character contradiction"
        ),
    }


def main() -> None:
    payload = build_payload()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
    for row in payload["rows"]:
        print(row["id"], row["rendered"])


if __name__ == "__main__":
    main()
