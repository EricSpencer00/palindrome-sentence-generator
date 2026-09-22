"""Repair one duplicated A-tub / Eh-but-a scaffold in the 644 child.

The edit is a bounded symmetric character window.  New event content is
selected as a left emission and exact reverse right obligation before the
surface is rendered; remaining repetition and grammar debt stay explicit.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.incumbent_498_deep_clause_transducer_20261002 import audit


PARENT = ROOT / "runs" / "incumbent-620-repeated-event-repair-20260922.json"
OUT = ROOT / "runs" / "incumbent-644-scaffold-repair-20260922.json"
PARENT_ID = "repeated-event-repair-mara-sees-644"
PARENT_SHA256 = "ff1839b066c8a5a8f5be3c5e95c4b5ce89a213f4e6878234cd6c748f0e980cc1"
OLD_LEFT = "A tub? He maps Aron"
OLD_RIGHT = "Nora, spam. Eh, but a"
NEW_LEFT = "Aidan draws maps, Aron"
NEW_RIGHT = "Nora, Spam's ward, Nadia."
WINDOW_LEFT_START = 115
WINDOW_LEFT_END = 129
SUPERSEDING_REPAIR_ARTIFACT = "runs/incumbent-652-clause-window-repair-20260922.json"
SUPERSEDING_REPAIR_ID = "clause-window-repair-nora-sees-666"
NEXT_OPERATOR = (
    "the 652 child is rejected/non-promoted for fragment debt; the completed "
    "same-window clause repair is in "
    f"{SUPERSEDING_REPAIR_ARTIFACT} row {SUPERSEDING_REPAIR_ID}; "
    "repair the clearest remaining grammar debt in the 666 child; use the "
    "clean 594 cursor-204/364 fallback only on contradiction"
)

FRONTIER = (
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
)


def normalize(text: str) -> str:
    return "".join(re.findall(r"[a-z]", text.lower()))


def independent_audit(text: str) -> dict[str, object]:
    tape = normalize(text)
    reverse = tape[::-1]
    forward_sha = hashlib.sha256(tape.encode()).hexdigest()
    reverse_sha = hashlib.sha256(reverse.encode()).hexdigest()
    return {
        "normalized_letters": len(tape),
        "two_pointer_exact": all(tape[i] == tape[-1 - i] for i in range(len(tape) // 2)),
        "sha256_forward": forward_sha,
        "sha256_reverse": reverse_sha,
        "sha_equal": forward_sha == reverse_sha,
    }


def validate_frontier_entry(entry: dict[str, object]) -> None:
    artifact = ROOT / str(entry["artifact"])
    assert artifact.exists(), artifact
    payload = json.loads(artifact.read_text())
    row = next(row for row in payload["rows"] if row["id"] == entry["id"])
    recomputed = independent_audit(str(row["rendered"]))
    assert recomputed["normalized_letters"] == entry["letters"]
    assert recomputed["two_pointer_exact"]
    assert recomputed["sha256_forward"] == entry["sha256"]
    assert recomputed["sha_equal"]


def raw_span_for_normalized_window(text: str, start: int, end: int) -> tuple[int, int]:
    indices = [index for index, char in enumerate(text) if "a" <= char.lower() <= "z"]
    return indices[start], indices[end - 1] + 1


def build_payload() -> dict[str, object]:
    parent_payload = json.loads(PARENT.read_text())
    parent = next(row for row in parent_payload["rows"] if row["id"] == PARENT_ID)
    parent_rendered = str(parent["rendered"])
    parent_tape = normalize(parent_rendered)
    parent_independent = independent_audit(parent_rendered)
    assert parent_independent["normalized_letters"] == 644
    assert parent_independent["two_pointer_exact"]
    assert parent_independent["sha256_forward"] == PARENT_SHA256
    assert parent_independent["sha_equal"]
    assert parent["audit"]["sha256_forward"] == PARENT_SHA256
    for frontier_entry in FRONTIER:
        validate_frontier_entry(frontier_entry)

    left_start, left_end = raw_span_for_normalized_window(
        parent_rendered, WINDOW_LEFT_START, WINDOW_LEFT_END
    )
    right_start, right_end = raw_span_for_normalized_window(
        parent_rendered,
        len(parent_tape) - WINDOW_LEFT_END,
        len(parent_tape) - WINDOW_LEFT_START,
    )
    assert parent_rendered[left_start:left_end] == OLD_LEFT
    assert parent_rendered[right_start:right_end] == OLD_RIGHT
    left_emission = normalize(NEW_LEFT)
    right_obligation = normalize(NEW_RIGHT)
    assert left_emission == right_obligation[::-1]

    rendered = (
        parent_rendered[:left_start]
        + NEW_LEFT
        + parent_rendered[left_end:right_start]
        + NEW_RIGHT
        + parent_rendered[right_end:]
    )
    project_audit = audit(rendered)
    independent = independent_audit(rendered)
    assert project_audit["letters"] == 652
    assert independent["normalized_letters"] == 652
    assert independent["two_pointer_exact"]
    assert independent["sha_equal"]
    assert project_audit["two_pointer_exact"]
    assert project_audit["byte_pointer_exact"]
    assert project_audit["project_validator_exact"]

    lowered = rendered.lower()
    assert lowered.count("a tub") == 1
    assert lowered.count("eh, but a") == 1

    row = {
        "id": "scaffold-repair-aidan-draws-652",
        "working_status": "rejected_non_promoted_for_fragment_debt",
        "rendered": rendered,
        "audit": project_audit,
        "independent_audit": independent,
        "parent_artifact": str(PARENT.relative_to(ROOT)),
        "parent_id": PARENT_ID,
        "parent_sha256": PARENT_SHA256,
        "growth_over_parent": independent["normalized_letters"] - 644,
        "new_event_content": [
            "Aidan draws maps, Aron",
            "Spam's ward, Nadia",
        ],
        "live_seam": {
            "normalized_window_left": [WINDOW_LEFT_START, WINDOW_LEFT_END],
            "normalized_window_right": [
                len(parent_tape) - WINDOW_LEFT_END,
                len(parent_tape) - WINDOW_LEFT_START,
            ],
            "old_left": OLD_LEFT,
            "old_right": OLD_RIGHT,
            "new_left": NEW_LEFT,
            "new_right": NEW_RIGHT,
            "initial_owner": "left_window",
            "left_emission": left_emission,
            "right_obligation": right_obligation,
            "right_consumption": right_obligation,
            "final_owner": None,
            "final_residual": "",
            "committed_character_contradictions": 0,
            "backtracks": 0,
        },
        "repetition_delta": {
            "A tub": {"before": 2, "after": 1},
            "Eh, but a": {"before": 2, "after": 1},
        },
        "grammar_debt": {
            "inherited_proper_palindromic_spans": True,
            "remaining_star_spam_scaffolding": True,
            "remaining_mara_stops_rats_scaffolding": 2,
            "rough_syntax_elsewhere": True,
            "human_reader_validation": False,
            "effect": "retain exact 644 lineage; do not demote 568",
        },
        "promotion_status": {
            "status": "rejected_non_promoted",
            "reason_code": "fragment_debt",
            "reason": (
                "The rendered Aidan-draws seam leaves a dangling vocative/appositive "
                "and a lowercase sentence start; it is exact evidence, not a "
                "promoted readable child."
            ),
            "superseded_by": {
                "artifact": SUPERSEDING_REPAIR_ARTIFACT,
                "id": SUPERSEDING_REPAIR_ID,
                "status": "completed_same_window_clause_repair",
            },
        },
        "next_operator": NEXT_OPERATOR,
        "provenance": (
            "smallest symmetric 14-letter window repair loaded from the exact "
            "644 child; new draw/ward event emissions close a live residual before admission"
        ),
    }

    return {
        "experiment_id": "incumbent-644-scaffold-repair-20260922",
        "method": "bounded symmetric A-tub/Eh-but-a scaffold repair with live residual ownership",
        "working_incumbent": {
            "artifact": "runs/incumbent-560-outer-causal-scene-20261002.json",
            "id": "outer-causal-scene-568-working-incumbent",
            "letters": 568,
            "sha256": "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380",
        },
        "stats": {
            "independently_exact_children": 1,
            "children_at_least_644": 1,
            "repaired_child_letters": independent["normalized_letters"],
            "window_letters_replaced": WINDOW_LEFT_END - WINDOW_LEFT_START,
            "committed_character_contradictions": 0,
            "backtracks": 0,
        },
        "preserved_frontier": list(FRONTIER),
        "rows": [row],
        "next_operator": NEXT_OPERATOR,
    }


def main() -> None:
    payload = build_payload()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
    print(payload["rows"][0]["rendered"])


if __name__ == "__main__":
    main()
