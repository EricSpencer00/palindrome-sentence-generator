"""Promote the exact 568-letter causal scene as the working length incumbent."""
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.incumbent_498_deep_clause_transducer_20261002 import audit
from experiments.incumbent_550_wide_center_story_20261002 import proper_spans
from llm_palindrome.admission import (
    has_distinct_content_words,
    has_repeated_nontrivial_unit,
    has_self_palindromic_proper_multiword_span,
    tokenize,
)


PARENT = ROOT / "runs" / "incumbent-550-central-event-bridge-20261002.json"
OUT = ROOT / "runs" / "incumbent-560-outer-causal-scene-20261002.json"
PARENT_SHA256 = "b5f98bfb0b44b31d8cbf78727672a74b588980e1fc8f1ff522a2c4ad1d800ccc"
EXPECTED_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
FRONTIER_556_SHA256 = "28b303081c7eeae9b0f4c7e274d71e73551c64f5ad389b2d992b6183597f6d14"

OLD_LEFT = "Nadia delivers maps. Nora stops rats. A tub? He maps Leon."
OLD_RIGHT = "Noel, spam. Eh, but a star spots Aron. Spam's reviled, Aidan."
NEW_LEFT = "Leon won. Wolf spots Nora. Nadia stops, so Tara rewards Nadia."
NEW_RIGHT = "Aidan's drawer, Aratos, spots Aidan. Aron stops flow now, Noel."


def build_payload() -> dict[str, object]:
    parent_payload = json.loads(PARENT.read_text())
    parent = parent_payload["rows"][0]
    assert parent["audit"]["sha256_forward"] == PARENT_SHA256
    parent_rendered = str(parent["rendered"])
    assert parent_rendered.count(OLD_LEFT) == 1
    assert parent_rendered.count(OLD_RIGHT) == 1

    rendered = parent_rendered.replace(OLD_LEFT, NEW_LEFT).replace(OLD_RIGHT, NEW_RIGHT)
    result_audit = audit(rendered)
    assert result_audit["letters"] == 568
    assert result_audit["sha256_forward"] == EXPECTED_SHA256
    assert result_audit["two_pointer_exact"]
    assert result_audit["byte_pointer_exact"]
    assert result_audit["project_validator_exact"]

    units = tokenize(rendered)
    spans = proper_spans(rendered)
    assert len(spans) == 65
    strict = {
        "proper_span_count": len(spans),
        "shortest_proper_span": spans[0],
        "no_self_palindromic_proper_multiword_span": not has_self_palindromic_proper_multiword_span(units),
        "no_repeated_nontrivial_unit": not has_repeated_nontrivial_unit(units),
        "distinct_content_words": has_distinct_content_words(units),
        "shortcut_clean": False,
        "human_certified": False,
        "status": (
            "exact working length incumbent; proper spans, repeated units, and "
            "rough prose are repair debt; human certification remains pending"
        ),
    }
    assert not strict["no_self_palindromic_proper_multiword_span"]
    assert not strict["no_repeated_nontrivial_unit"]
    assert not strict["distinct_content_words"]

    return {
        "experiment_id": "incumbent-560-outer-causal-scene-20261002",
        "method": "replace one complete outer macro with a boundary-shifting causal scene",
        "parent": {
            "artifact": str(PARENT.relative_to(ROOT)),
            "letters": 560,
            "sha256": PARENT_SHA256,
            "working_track_only": True,
        },
        "stats": {
            "independently_exact_children": 1,
            "longest_letters": 568,
            "working_length_incumbent_letters": 568,
            "shortcut_clean_children": 0,
            "proper_span_count": len(spans),
        },
        "working_length_incumbent": {
            "artifact": str(OUT.relative_to(ROOT)),
            "id": "outer-causal-scene-568-working-incumbent",
            "letters": 568,
            "sha256": EXPECTED_SHA256,
        },
        "diverse_repair_frontier": [
            {
                "artifact": str(PARENT.relative_to(ROOT)),
                "id": "central-distinct-events-560",
                "letters": 560,
                "sha256": PARENT_SHA256,
            },
            {
                "artifact": "runs/incumbent-498-event-frame-seam-repair-20261002.json",
                "id": "depth39-longest-f1g1h1r",
                "letters": 556,
                "sha256": FRONTIER_556_SHA256,
            },
        ],
        "rows": [{
            "id": "outer-causal-scene-568-working-incumbent",
            "working_status": "working_length_incumbent",
            "rendered": rendered,
            "audit": result_audit,
            "parent_artifact": str(PARENT.relative_to(ROOT)),
            "parent_sha256": PARENT_SHA256,
            "source_498_artifact": "runs/overhang-growth-from-240-20261001.json",
            "source_498_sha256": "e809a2a05a414347615f68f27f6b4974aa00ede287fdb2a4b23f6ffbadec9032",
            "growth_over_parent": 8,
            "replacement": {
                "old_left": OLD_LEFT,
                "old_right": OLD_RIGHT,
                "new_left": NEW_LEFT,
                "new_right": NEW_RIGHT,
                "new_event_content": [
                    "Leon wins",
                    "a wolf spots Nora",
                    "Tara rewards Nadia because Nadia stops",
                    "Aron stops a flow",
                ],
            },
            "live_state": {
                "residual_sequence": ["tara", "s", "nadia"],
                "boundary_shifts": ["so|Tara / Aratos", "reward|s / Aidan's"],
                "final_residual": "",
            },
            "strict_admission": strict,
            "repair_debt": {
                "proper_palindromic_spans": len(spans),
                "repeated_nontrivial_units": True,
                "rough_prose": True,
                "effect": "prioritize repair; do not reject the exact working result",
            },
            "next_operator": (
                "reopen an actual partial-word seam of this 568-letter tape, "
                "carry residual ownership across it, save independently exact "
                "children with new event content longer than 568, then repair "
                "the worst repeated outer shell"
            ),
        }],
    }


def main() -> None:
    payload = build_payload()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
    print(payload["rows"][0]["rendered"])


if __name__ == "__main__":
    main()
