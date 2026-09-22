"""Promote a bounded 666 readability frontier from the causal parent.

The result is the active 666 readability frontier, not the working incumbent:
the 568 child remains incumbent, and the causal 666 parent remains retained as
comparison evidence because this edit removes a good causal clause.
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


PARENT = ROOT / "runs" / "incumbent-666-causal-window-repair-20260922.json"
OUT = ROOT / "runs" / "incumbent-666-comparison-alternative-20260922.json"
PARENT_ID = "causal-window-repair-nadia-saw-666"
PARENT_SHA256 = "268012039a4f838fdd86b15f37e4bd6d554208f6dd7fd4ceb8fb01c26776b015"
CHILD_SHA256 = "cafd77235f82d9ff4f68814dc7e03d196bf719bf1ec5d541e172073502e12297"
WINDOW_LEFT = (20, 64)
WINDOW_RIGHT = (602, 646)
RAW_LEFT = (26, 82)
RAW_RIGHT = (818, 877)
OLD_LEFT = " Nadia stops, so Tara rewards Nadia. Nora delivers maps."
OLD_RIGHT = " Spam's reviled, Aron. Aidan's drawer, Aratos, spots Aidan."
NEW_LEFT = "Nora sees Nadia. Nadia sees Ira. Sara saw God. Ari saw Dog."
NEW_RIGHT = "God was Ira. Dog was Aras. Ari sees Aidan. Aidan sees Aron."

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
    for frontier_entry in FRONTIER:
        validate_frontier_entry(frontier_entry)

    assert parent_tape[WINDOW_LEFT[0] : WINDOW_LEFT[1]] == normalize(OLD_LEFT)
    assert parent_tape[WINDOW_RIGHT[0] : WINDOW_RIGHT[1]] == normalize(OLD_RIGHT)
    assert parent_rendered[RAW_LEFT[0] : RAW_LEFT[1]] == OLD_LEFT
    assert parent_rendered[RAW_RIGHT[0] : RAW_RIGHT[1]] == OLD_RIGHT
    left_emission = normalize(NEW_LEFT)
    right_obligation = normalize(NEW_RIGHT)
    assert len(left_emission) == WINDOW_LEFT[1] - WINDOW_LEFT[0] == 44
    assert len(right_obligation) == WINDOW_RIGHT[1] - WINDOW_RIGHT[0] == 44
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
    assert OLD_RIGHT not in rendered
    assert "Spam's reviled, Nadia" not in rendered
    for clause in (
        "Nora sees Nadia.",
        "Nadia sees Ira.",
        "Sara saw God.",
        "Ari saw Dog.",
        "God was Ira.",
        "Dog was Aras.",
        "Ari sees Aidan.",
        "Aidan sees Aron.",
    ):
        assert clause in rendered

    row = {
        "id": "comparison-alternative-nora-sees-666",
        "working_status": "active_666_readability_frontier",
        "promotion_status": {
            "promoted": True,
            "status": "promoted_active_readability_frontier",
            "working_incumbent_unchanged": True,
            "rationale": (
                "The complete finite-clause surface removes the targeted right-cluster "
                "fragment while preserving exact 666 length and empty residual ownership."
            ),
            "tradeoff": (
                "This edit removes a good causal clause from the parent, so the causal "
                "666 artifact remains retained as comparison evidence; it does not "
                "replace the 568 working incumbent."
            ),
            "comparison_retained": {
                "artifact": str(PARENT.relative_to(ROOT)),
                "id": PARENT_ID,
                "sha256": PARENT_SHA256,
            },
        },
        "rendered": rendered,
        "audit": project_audit,
        "independent_audit": independent,
        "parent_artifact": str(PARENT.relative_to(ROOT)),
        "parent_id": PARENT_ID,
        "parent_sha256": PARENT_SHA256,
        "growth_over_parent": independent["normalized_letters"] - 666,
        "new_event_content": [
            "Nora sees Nadia",
            "Nadia sees Ira",
            "Sara saw God",
            "Ari saw Dog",
            "God was Ira",
            "Dog was Aras",
            "Ari sees Aidan",
            "Aidan sees Aron",
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
            "initial_owner": "left_comparison_window",
            "left_emission": left_emission,
            "right_obligation": right_obligation,
            "right_consumption": right_obligation,
            "final_owner": None,
            "final_residual": "",
            "committed_character_contradictions": 0,
            "backtracks": 0,
        },
        "readability_delta": {
            "targeted_right_cluster_removed": True,
            "targeted_spams_reviled_nadia_removed": True,
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
            "effect": "active readability frontier; do not demote 568; retain causal 666 comparison",
        },
        "provenance": (
            "reviewer-derived 44-letter mirrored replacement loaded from the exact "
            "causal 666 child; promoted for readability while retaining the causal "
            "parent as comparison evidence"
        ),
    }

    return {
        "experiment_id": "incumbent-666-comparison-alternative-20260922",
        "method": "bounded 44-letter mirrored comparison alternative with live residual ownership",
        "active_readability_frontier": {
            "artifact": str(OUT.relative_to(ROOT)),
            "id": "comparison-alternative-nora-sees-666",
            "letters": 666,
            "sha256": CHILD_SHA256,
            "promoted": True,
            "working_incumbent_unchanged": True,
        },
        "comparison_retained": {
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
            "left_window_letters": 44,
            "right_window_letters": 44,
            "committed_character_contradictions": 0,
            "backtracks": 0,
        },
        "preserved_frontier": list(FRONTIER),
        "rows": [row],
        "next_operator": (
            "repair the next actual stop/spot seam at normalized [64,127)/[539,602), "
            "raw [86,170)/[739,822); retain 568 as working incumbent"
        ),
    }


def main() -> None:
    payload = build_payload()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
    print(payload["rows"][0]["rendered"])


if __name__ == "__main__":
    main()
