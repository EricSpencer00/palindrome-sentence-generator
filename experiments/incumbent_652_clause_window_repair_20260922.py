"""Repair the inherited 652 window with complete mirrored clauses.

This is the same semantic window opened at [115,129)/[515,529) on the 644
parent.  Its 652 expansion is [115,133)/[519,537); only that lineage window is
edited, and the new clauses carry a live residual before exact admission.
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


PARENT = ROOT / "runs" / "incumbent-644-scaffold-repair-20260922.json"
OUT = ROOT / "runs" / "incumbent-652-clause-window-repair-20260922.json"
PARENT_ID = "scaffold-repair-aidan-draws-652"
PARENT_SHA256 = "0b7f1950beac54b34a8ec66e5150f995abeda4d2d07bfd2f0eb454fed108f1e5"
OLD_LEFT = "Aidan draws maps, Aron."
OLD_RIGHT = "Nora, Spam's ward, Nadia."
NEW_LEFT = "Nora sees Aram. Mara sees Nadia."
NEW_RIGHT = "Aidan sees Aram. Mara sees Aron;"
WINDOW_LEFT_START = 115
WINDOW_LEFT_END = 133
LINEAGE_PARENT_WINDOW = {
    "artifact": "runs/incumbent-620-repeated-event-repair-20260922.json",
    "normalized_left": [115, 129],
    "normalized_right": [515, 529],
}

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
    raw_end = indices[end - 1] + 1
    if raw_end < len(text) and text[raw_end] == ".":
        raw_end += 1
    return indices[start], raw_end


def build_payload() -> dict[str, object]:
    parent_payload = json.loads(PARENT.read_text())
    parent = next(row for row in parent_payload["rows"] if row["id"] == PARENT_ID)
    parent_rendered = str(parent["rendered"])
    parent_tape = normalize(parent_rendered)
    parent_independent = independent_audit(parent_rendered)
    assert parent_independent["normalized_letters"] == 652
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
    assert project_audit["letters"] == 666
    assert independent["normalized_letters"] == 666
    assert independent["two_pointer_exact"]
    assert independent["sha_equal"]
    assert project_audit["two_pointer_exact"]
    assert project_audit["byte_pointer_exact"]
    assert project_audit["project_validator_exact"]
    assert "Aidan draws maps, Aron" not in rendered
    assert "Nora, Spam's ward, Nadia" not in rendered
    assert "Nora sees Aram. Mara sees Nadia." in rendered
    assert "Aidan sees Aram. Mara sees Aron; star spots Aron" in rendered

    row = {
        "id": "clause-window-repair-nora-sees-666",
        "working_status": "652_lineage_repair_frontier",
        "rendered": rendered,
        "audit": project_audit,
        "independent_audit": independent,
        "parent_artifact": str(PARENT.relative_to(ROOT)),
        "parent_id": PARENT_ID,
        "parent_sha256": PARENT_SHA256,
        "growth_over_parent": independent["normalized_letters"] - 652,
        "new_event_content": [
            "Nora sees Aram",
            "Mara sees Nadia",
            "Aidan sees Aram",
            "Mara sees Aron",
        ],
        "live_seam": {
            "lineage_parent_window": LINEAGE_PARENT_WINDOW,
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
        "syntax_repair": {
            "complete_left_clauses": True,
            "complete_right_clauses": True,
            "dangling_vocative_or_appositive": False,
            "predicate_less_fragment": False,
            "sentence_capitalization": "complete clauses begin with Nora/Aidan/Mara; semicolon joins the existing star clause",
        },
        "grammar_debt": {
            "inherited_proper_palindromic_spans": True,
            "remaining_a_tub_repetition": 1,
            "remaining_star_spam_scaffolding": True,
            "remaining_mara_stops_rats_scaffolding": 2,
            "rough_syntax_elsewhere": True,
            "human_reader_validation": False,
            "effect": "retain exact 666 lineage; do not demote 568",
        },
        "provenance": (
            "same lineage window repair loaded from the exact 652 child; complete "
            "mirrored clauses consume a live residual before independent admission"
        ),
    }

    return {
        "experiment_id": "incumbent-652-clause-window-repair-20260922",
        "method": "same-window complete-clause repair with live residual ownership",
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
            "complete_mirrored_clauses": 4,
            "committed_character_contradictions": 0,
            "backtracks": 0,
        },
        "preserved_frontier": list(FRONTIER),
        "rows": [row],
        "next_operator": (
            "repair the clearest remaining grammar debt in the 666 child; "
            "change this same window only on a committed contradiction"
        ),
    }


def main() -> None:
    payload = build_payload()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
    print(payload["rows"][0]["rendered"])


if __name__ == "__main__":
    main()
