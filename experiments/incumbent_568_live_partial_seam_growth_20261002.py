"""Grow the 568 incumbent through a live partial-word seam, then repair a shell."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.incumbent_498_deep_clause_transducer_20261002 import audit
from experiments.incumbent_550_wide_center_story_20261002 import proper_spans


PARENT = ROOT / "runs" / "incumbent-560-outer-causal-scene-20261002.json"
OUT = ROOT / "runs" / "incumbent-568-live-partial-seam-growth-20261002.json"
PARENT_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"

SEAMS = [
    {
        "id": "live-leon-seam-590",
        "left": "A ram saw Leon. Le",
        "right": "el. Noel was Mara.",
        "event": "a ram saw Leon",
        "letters": 590,
        "sha256": "293f5ef2d0bd9d794f1e3b90fc00c66db80ac127d3665208a666814acd7fd112",
    },
    {
        "id": "live-nora-seam-590",
        "left": "A ram saw Nora. Le",
        "right": "el. Aron was Mara.",
        "event": "a ram saw Nora",
        "letters": 590,
        "sha256": "6a009952555a29d40dd9281220b8bc92941f9a11594703e25fd7d7c774cec540",
    },
    {
        "id": "live-nadia-seam-592",
        "left": "A ram saw Nadia. Le",
        "right": "el. Aidan was Mara.",
        "event": "a ram saw Nadia",
        "letters": 592,
        "sha256": "f27b755155af851991a9cd1d47341784e209b4280ae24c1ffd46daf3e7519391",
    },
]

OLD_LEFT = "Mara stops rats. A tub? He maps Aron."
OLD_RIGHT = "Nora, spam. Eh, but a star spots Aram."
NEW_LEFT = "Mara spots rats. Leon maps Nora. Draw no maps."
NEW_RIGHT = "Spam onward. Aron, spam Noel. Star stops Aram."
REPAIRED_SHA256 = "eacd84ecc82aacb84fe19b557faf91494370e86c986248be5e37de9ad7298248"


def normalize(text: str) -> str:
    return "".join(re.findall(r"[a-z]", text.lower()))


def build_payload() -> dict[str, object]:
    parent_payload = json.loads(PARENT.read_text())
    parent = parent_payload["rows"][0]
    assert parent["audit"]["sha256_forward"] == PARENT_SHA256
    assert parent["audit"]["letters"] == 568
    parent_rendered = str(parent["rendered"])
    parent_tape = normalize(parent_rendered)
    assert parent_tape == parent_tape[::-1]
    assert parent_tape[:4] == "leon" and parent_tape[-4:] == "noel"

    # Reopen the first/last word after two committed letters on each side.
    words = parent_rendered.split()
    assert words[0] == "Leon" and words[-1] == "Noel."
    retained_rendered = "on " + " ".join(words[1:-1]) + " No"
    retained_tape = normalize(retained_rendered)
    assert retained_tape == parent_tape[2:-2]
    assert retained_tape == retained_tape[::-1]

    rows: list[dict[str, object]] = []
    rendered_by_id: dict[str, str] = {}
    for seam in SEAMS:
        left = str(seam["left"])
        right = str(seam["right"])
        assert normalize(left) == normalize(right)[::-1]
        rendered = left + retained_rendered + right
        result_audit = audit(rendered)
        assert result_audit["letters"] == seam["letters"]
        assert result_audit["sha256_forward"] == seam["sha256"]
        assert result_audit["two_pointer_exact"]
        assert result_audit["byte_pointer_exact"]
        assert result_audit["project_validator_exact"]
        rendered_by_id[str(seam["id"])] = rendered
        rows.append({
            "id": seam["id"],
            "working_status": "diverse_exact_growth_frontier",
            "rendered": rendered,
            "audit": result_audit,
            "parent_artifact": str(PARENT.relative_to(ROOT)),
            "parent_sha256": PARENT_SHA256,
            "growth_over_parent": int(seam["letters"]) - 568,
            "new_event_content": [seam["event"]],
            "live_seam": {
                "parent_left_cursor": 2,
                "parent_right_cursor_exclusive": 566,
                "retained_letters": 564,
                "left_partial_join": "Le|on",
                "right_partial_join": "No|el",
                "initial_owner": "left",
                "left_emission": normalize(left),
                "right_obligation": normalize(left)[::-1],
                "right_consumption": normalize(right),
                "final_owner": None,
                "final_residual": "",
                "backtracks": 0,
            },
            "provenance": "deterministic authored seam equation over the verified 568 tape",
        })

    repair_parent_id = "live-nadia-seam-592"
    repair_parent = rendered_by_id[repair_parent_id]
    assert repair_parent.count(OLD_LEFT) == 1
    assert repair_parent.count(OLD_RIGHT) == 1
    assert normalize(NEW_LEFT) == normalize(NEW_RIGHT)[::-1]
    repaired = repair_parent.replace(OLD_LEFT, NEW_LEFT, 1).replace(OLD_RIGHT, NEW_RIGHT, 1)
    repaired_audit = audit(repaired)
    assert repaired_audit["letters"] == 608
    assert repaired_audit["sha256_forward"] == REPAIRED_SHA256
    assert repaired_audit["two_pointer_exact"]
    assert repaired_audit["byte_pointer_exact"]
    assert repaired_audit["project_validator_exact"]

    repeated_phrases = ["Mara stops rats.", "A tub?", "Eh, but a star spots Aram."]
    repetition_delta = {
        phrase: {
            "before": repair_parent.count(phrase),
            "after": repaired.count(phrase),
        }
        for phrase in repeated_phrases
    }
    assert all(delta == {"before": 2, "after": 1} for delta in repetition_delta.values())

    repaired_spans = proper_spans(repaired)
    rows.append({
        "id": "live-nadia-seam-shell-repair-608",
        "working_status": "working_length_incumbent",
        "rendered": repaired,
        "audit": repaired_audit,
        "parent_artifact": str(PARENT.relative_to(ROOT)),
        "parent_sha256": PARENT_SHA256,
        "repair_parent_id": repair_parent_id,
        "growth_over_568_parent": 40,
        "growth_over_592_repair_parent": 16,
        "new_event_content": [
            "a ram saw Nadia",
            "Mara spots rats",
            "Leon maps Nora",
            "draw no maps",
            "carry maps onward",
            "a star stops Aram",
        ],
        "live_seam": rows[-1]["live_seam"],
        "shell_repair": {
            "old_left": OLD_LEFT,
            "old_right": OLD_RIGHT,
            "new_left": NEW_LEFT,
            "new_right": NEW_RIGHT,
            "equation_exact": True,
            "repetition_delta": repetition_delta,
        },
        "repair_debt": {
            "proper_palindromic_spans": len(repaired_spans),
            "rough_prose": True,
            "human_certified": False,
            "effect": "continue repairing the worst remaining shell; do not reject exact growth",
        },
        "provenance": "deterministic authored live-seam growth followed by one mirrored shell repair",
        "next_operator": (
            "retain this 608-letter exact tape and reopen its next highest-impact "
            "repeated shell; backtrack only if the normalized character equations contradict"
        ),
    })

    return {
        "experiment_id": "incumbent-568-live-partial-seam-growth-20261002",
        "method": "partial-word residual ownership plus mirrored outer-shell repair",
        "parent": {
            "artifact": str(PARENT.relative_to(ROOT)),
            "id": parent["id"],
            "letters": 568,
            "sha256": PARENT_SHA256,
        },
        "stats": {
            "independently_exact_children": len(rows),
            "children_longer_than_parent": len(rows),
            "longest_letters": 608,
            "committed_character_contradictions": 0,
            "backtracks": 0,
            "repeated_phrases_reduced": len(repetition_delta),
        },
        "working_length_incumbent": {
            "artifact": str(OUT.relative_to(ROOT)),
            "id": "live-nadia-seam-shell-repair-608",
            "letters": 608,
            "sha256": REPAIRED_SHA256,
        },
        "preserved_frontier": [
            {
                "artifact": str(PARENT.relative_to(ROOT)),
                "id": parent["id"],
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
                "artifact": "runs/incumbent-498-event-frame-seam-repair-20261002.json",
                "id": "depth39-longest-f1g1h1r",
                "letters": 556,
                "sha256": "28b303081c7eeae9b0f4c7e274d71e73551c64f5ad389b2d992b6183597f6d14",
            },
        ],
        "rows": rows,
    }


def main() -> None:
    payload = build_payload()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
    print(payload["rows"][-1]["rendered"])


if __name__ == "__main__":
    main()
