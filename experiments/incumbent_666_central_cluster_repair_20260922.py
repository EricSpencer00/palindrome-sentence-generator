"""Repair the 666 child's repeated central event scaffold.

The bounded edit replaces the actual centered normalized window [297,369) in
the exact 666 child.  The left half emits complete finite clauses and the
right half consumes its character-reversed obligation before admission.
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


PARENT = ROOT / "runs" / "incumbent-652-clause-window-repair-20260922.json"
OUT = ROOT / "runs" / "incumbent-666-central-cluster-repair-20260922.json"
PARENT_ID = "clause-window-repair-nora-sees-666"
PARENT_SHA256 = "b6ccddf4d34f9d2705238f615336e633cb76bccdd010913d44799881df75584e"
CHILD_SHA256 = "3e8792786eb9a5f406251e2071650d7b0b53e1979cb1786df4d510ab9ff00d40"
WINDOW_START = 297
WINDOW_SEAM = 333
WINDOW_END = 369
OLD_WINDOW = (
    "Nadia delivers maps. Leon. Ari delivers maps. Spam's reviled, Ira. "
    "Noel; spam's reviled, Aidan"
)
NEW_LEFT = "Nadia sees Mara. Nora sees Nadia. Ari sees God."
NEW_RIGHT = "Dog sees Ira. Aidan sees Aron. Aram sees Aidan"

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
        "two_pointer_exact": all(
            tape[index] == tape[-1 - index] for index in range(len(tape) // 2)
        ),
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
    assert parent_independent["normalized_letters"] == 666
    assert parent_independent["two_pointer_exact"]
    assert parent_independent["sha256_forward"] == PARENT_SHA256
    assert parent_independent["sha_equal"]
    assert parent["audit"]["sha256_forward"] == PARENT_SHA256
    for frontier_entry in FRONTIER:
        validate_frontier_entry(frontier_entry)

    left_start, right_end = raw_span_for_normalized_window(
        parent_rendered, WINDOW_START, WINDOW_END
    )
    assert parent_rendered[left_start:right_end] == OLD_WINDOW
    left_emission = normalize(NEW_LEFT)
    right_obligation = normalize(NEW_RIGHT)
    assert len(left_emission) == WINDOW_SEAM - WINDOW_START
    assert len(right_obligation) == WINDOW_END - WINDOW_SEAM
    assert left_emission == right_obligation[::-1]
    assert left_emission + right_obligation == (
        left_emission + right_obligation
    )[::-1]

    rendered = (
        parent_rendered[:left_start]
        + NEW_LEFT
        + " "
        + NEW_RIGHT
        + parent_rendered[right_end:]
    )
    project_audit = audit(rendered)
    independent = independent_audit(rendered)
    assert project_audit["letters"] == 666
    assert independent["normalized_letters"] == 666
    assert independent["two_pointer_exact"]
    assert independent["sha256_forward"] == CHILD_SHA256
    assert independent["sha_equal"]
    assert project_audit["two_pointer_exact"]
    assert project_audit["byte_pointer_exact"]
    assert project_audit["project_validator_exact"]
    assert "Nadia delivers maps. Leon. Ari delivers maps." not in rendered
    assert "Spam's reviled, Ira. Noel; spam's reviled, Aidan" not in rendered
    for clause in (
        "Nadia sees Mara.",
        "Nora sees Nadia.",
        "Ari sees God.",
        "Dog sees Ira.",
        "Aidan sees Aron.",
        "Aram sees Aidan.",
    ):
        assert clause in rendered

    row = {
        "id": "central-cluster-repair-nadia-sees-666",
        "working_status": "666_lineage_repair_frontier",
        "rendered": rendered,
        "audit": project_audit,
        "independent_audit": independent,
        "parent_artifact": str(PARENT.relative_to(ROOT)),
        "parent_id": PARENT_ID,
        "parent_sha256": PARENT_SHA256,
        "growth_over_parent": independent["normalized_letters"] - 666,
        "new_event_content": [
            "Nadia sees Mara",
            "Nora sees Nadia",
            "Ari sees God",
            "Dog sees Ira",
            "Aidan sees Aron",
            "Aram sees Aidan",
        ],
        "live_seam": {
            "normalized_window_left": [WINDOW_START, WINDOW_SEAM],
            "normalized_window_right": [WINDOW_SEAM, WINDOW_END],
            "old_window": OLD_WINDOW,
            "new_left": NEW_LEFT,
            "new_right": NEW_RIGHT,
            "initial_owner": "left_central_window",
            "left_emission": left_emission,
            "right_obligation": right_obligation,
            "right_consumption": right_obligation,
            "final_owner": None,
            "final_residual": "",
            "committed_character_contradictions": 0,
            "backtracks": 0,
        },
        "readability_delta": {
            "central_delivers_maps_before": 2,
            "central_delivers_maps_after": 0,
            "central_spams_reviled_before": 2,
            "central_spams_reviled_after": 0,
            "complete_finite_clauses": True,
            "predicate_less_fragment": False,
            "dangling_vocative_or_appositive": False,
        },
        "grammar_debt": {
            "inherited_proper_palindromic_spans": True,
            "remaining_delivers_maps_elsewhere": True,
            "remaining_star_spam_scaffolding": True,
            "remaining_mara_stops_rats_scaffolding": 2,
            "rough_syntax_elsewhere": True,
            "human_reader_validation": False,
            "effect": "retain exact 666 lineage; do not demote 568",
        },
        "provenance": (
            "bounded centered window loaded from the exact 666 child; six new "
            "finite event clauses consume the live reverse-character obligation "
            "before independent admission"
        ),
    }

    return {
        "experiment_id": "incumbent-666-central-cluster-repair-20260922",
        "method": "bounded central repeated-event repair with live residual ownership",
        "working_incumbent": {
            "artifact": "runs/incumbent-560-outer-causal-scene-20261002.json",
            "id": "outer-causal-scene-568-working-incumbent",
            "letters": 568,
            "sha256": "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380",
        },
        "stats": {
            "independently_exact_children": 1,
            "children_at_least_666": 1,
            "repaired_child_letters": independent["normalized_letters"],
            "central_window_letters_before": WINDOW_END - WINDOW_START,
            "central_window_letters_after": len(left_emission) + len(right_obligation),
            "committed_character_contradictions": 0,
            "backtracks": 0,
        },
        "preserved_frontier": list(FRONTIER),
        "rows": [row],
        "next_operator": (
            "repair the clearest remaining grammar debt in the 666 child; "
            "change this central window only on a committed contradiction"
        ),
    }


def main() -> None:
    payload = build_payload()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
    print(payload["rows"][0]["rendered"])


if __name__ == "__main__":
    main()
