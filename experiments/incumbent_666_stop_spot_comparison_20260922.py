"""Record a bounded stop/spot comparison child from the active 666 frontier.

The reviewer-specified 63-letter halves replace the exact repeated stop/spot
shell, but remain pending full-text readability review and are not promoted.
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


PARENT = ROOT / "runs" / "incumbent-666-comparison-alternative-20260922.json"
OUT = ROOT / "runs" / "incumbent-666-stop-spot-comparison-20260922.json"
PARENT_ID = "comparison-alternative-nora-sees-666"
PARENT_SHA256 = "cafd77235f82d9ff4f68814dc7e03d196bf719bf1ec5d541e172073502e12297"
CHILD_SHA256 = "484cd29ca3dafc3baab18f7d5ac7788cd941be5e9450f47f4038d2dbb217004b"
WINDOW_LEFT = (64, 127)
WINDOW_RIGHT = (539, 602)
RAW_LEFT = (86, 170)
RAW_RIGHT = (739, 822)
OLD_LEFT = " Mara stops rats. Nora spots a ram. Mara sees rats. Nora stops rats. Nora sees Aram."
OLD_RIGHT = " Mara sees Aron; star spots Aron. Star sees Aram. Mara stops Aron. Star spots Aram."
NEW_LEFT = "Mara saw Noel live. Nadia stops Aram. Nadia sees Aram. Aidan sees Ira. Ira saw Dog."
NEW_RIGHT = "God was Ari. Ari sees Nadia. Mara sees Aidan. Mara spots Aidan. Evil Leon was Aram."

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
    assert parent["promotion_status"]["promoted"] is True
    for frontier_entry in FRONTIER:
        validate_frontier_entry(frontier_entry)

    assert parent_tape[WINDOW_LEFT[0] : WINDOW_LEFT[1]] == normalize(OLD_LEFT)
    assert parent_tape[WINDOW_RIGHT[0] : WINDOW_RIGHT[1]] == normalize(OLD_RIGHT)
    assert parent_rendered[RAW_LEFT[0] : RAW_LEFT[1]] == OLD_LEFT
    assert parent_rendered[RAW_RIGHT[0] : RAW_RIGHT[1]] == OLD_RIGHT
    left_emission = normalize(NEW_LEFT)
    right_obligation = normalize(NEW_RIGHT)
    assert len(left_emission) == WINDOW_LEFT[1] - WINDOW_LEFT[0] == 63
    assert len(right_obligation) == WINDOW_RIGHT[1] - WINDOW_RIGHT[0] == 63
    assert left_emission == right_obligation[::-1]

    rendered = (
        parent_rendered[: RAW_LEFT[0]]
        + " "
        + NEW_LEFT
        + parent_rendered[RAW_LEFT[1] : RAW_RIGHT[0]]
        + " "
        + NEW_RIGHT
        + parent_rendered[RAW_RIGHT[1] :]
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
    assert OLD_LEFT not in rendered
    assert OLD_RIGHT not in rendered
    for clause in (
        "Mara saw Noel live.",
        "Nadia stops Aram.",
        "Nadia sees Aram.",
        "Aidan sees Ira.",
        "Ira saw Dog.",
        "God was Ari.",
        "Ari sees Nadia.",
        "Mara sees Aidan.",
        "Mara spots Aidan.",
        "Evil Leon was Aram.",
    ):
        assert clause in rendered

    row = {
        "id": "stop-spot-comparison-mara-saw-666",
        "working_status": "comparison_frontier_alternative",
        "promotion_status": {
            "promoted": False,
            "status": "pending_full_text_readability_review",
            "reason": "Bounded comparison child; do not auto-promote over the active 666 frontier.",
        },
        "rendered": rendered,
        "audit": project_audit,
        "independent_audit": independent,
        "parent_artifact": str(PARENT.relative_to(ROOT)),
        "parent_id": PARENT_ID,
        "parent_sha256": PARENT_SHA256,
        "growth_over_parent": independent["normalized_letters"] - 666,
        "new_event_content": [
            "Mara saw Noel live",
            "Nadia stops Aram",
            "Nadia sees Aram",
            "Aidan sees Ira",
            "Ira saw Dog",
            "God was Ari",
            "Ari sees Nadia",
            "Mara sees Aidan",
            "Mara spots Aidan",
            "Evil Leon was Aram",
        ],
        "live_seam": {
            "normalized_window_left": list(WINDOW_LEFT),
            "normalized_window_right": list(WINDOW_RIGHT),
            "raw_window_left": list(RAW_LEFT),
            "raw_window_right": list(RAW_RIGHT),
            "old_left": OLD_LEFT,
            "old_right": OLD_RIGHT,
            "new_left": NEW_LEFT,
            "new_right": NEW_RIGHT,
            "initial_owner": "left_stop_spot_comparison_window",
            "left_emission": left_emission,
            "right_obligation": right_obligation,
            "right_consumption": right_obligation,
            "final_owner": None,
            "final_residual": "",
            "committed_character_contradictions": 0,
            "backtracks": 0,
        },
        "readability_delta": {
            "targeted_old_stop_spot_cluster_removed": True,
            "complete_finite_clauses": True,
            "predicate_less_fragment": False,
            "dangling_vocative_or_appositive": False,
        },
        "grammar_debt": {
            "inherited_proper_palindromic_spans": True,
            "remaining_delivers_maps_elsewhere": True,
            "remaining_star_spam_scaffolding": True,
            "remaining_mara_stops_rats_scaffolding": True,
            "rough_syntax_elsewhere": True,
            "human_reader_validation": False,
            "effect": "retain as comparison alternative; do not demote active 666 or 568",
        },
        "provenance": (
            "reviewer-derived 63-letter mirrored stop/spot replacement loaded from "
            "the promoted exact 666 frontier; pending full-text review"
        ),
    }

    return {
        "experiment_id": "incumbent-666-stop-spot-comparison-20260922",
        "method": "bounded 63-letter mirrored stop/spot comparison with live residual ownership",
        "active_frontier_parent": {
            "artifact": str(PARENT.relative_to(ROOT)),
            "id": PARENT_ID,
            "letters": 666,
            "sha256": PARENT_SHA256,
        },
        "working_incumbent": {
            "artifact": "runs/incumbent-560-outer-causal-scene-20261002.json",
            "id": "outer-causal-scene-568-working-incumbent",
            "letters": 568,
            "sha256": "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380",
        },
        "stats": {
            "independently_exact_children": 1,
            "children_at_least_666": 1,
            "comparison_child_letters": independent["normalized_letters"],
            "left_window_letters": 63,
            "right_window_letters": 63,
            "committed_character_contradictions": 0,
            "backtracks": 0,
        },
        "preserved_frontier": list(FRONTIER),
        "rows": [row],
        "next_operator": (
            "perform full-text readability review of this stop/spot comparison; "
            "do not promote without that review"
        ),
    }


def main() -> None:
    payload = build_payload()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
    print(payload["rows"][0]["rendered"])


if __name__ == "__main__":
    main()
