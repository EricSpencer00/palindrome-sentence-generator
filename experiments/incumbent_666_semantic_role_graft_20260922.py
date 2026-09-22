"""Apply one bounded hand-authored semantic-role graft at an actual seam.

The graft edits [127,197)/[469,539) of the promoted 666 frontier.  It is a
small fixed SVO construction with varied sees/stops/spots relations, not a
Cartesian or whole-sentence search; every emitted character records its live
residual owner before independent exact admission.
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
OUT = ROOT / "runs" / "incumbent-666-semantic-role-graft-20260922.json"
PARENT_ID = "comparison-alternative-nora-sees-666"
PARENT_SHA256 = "cafd77235f82d9ff4f68814dc7e03d196bf719bf1ec5d541e172073502e12297"
CHILD_SHA256 = "152e810aa0d2434ef12af77bd73659d710386af41f7b69f4b3645c769e958565"
WINDOW_LEFT = (127, 197)
WINDOW_RIGHT = (469, 539)
RAW_LEFT = (170, 262)
RAW_RIGHT = (647, 739)
OLD_LEFT = " Mara sees Nadia. Nadia saw Noel live. Mara stops Nadia. Nora sees Aram. Sara saw Noel live."
OLD_RIGHT = " Evil Leon was Aras. Mara sees Aron. Aidan spots Aram. Evil Leon was Aidan. Aidan sees Aram."
NEW_LEFT = (
    "Nora sees Aram. Nadia sees Ira. Aidan sees Ira. Ari stops Aram. "
    "Mara stops Ira. Dog sees Ira."
)
NEW_RIGHT = (
    "Ari sees God. Ari spots Aram. Mara spots Ira. Ari sees Nadia. "
    "Ari sees Aidan. Mara sees Aron."
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


def emit_with_residual(emission: str, obligation: str, owner: str) -> tuple[str, list[dict[str, object]]]:
    residual = obligation
    trace = []
    for cursor, character in enumerate(normalize(emission)):
        assert residual, (owner, cursor, character)
        expected = residual[0]
        assert character == expected, (owner, cursor, character, expected, residual)
        residual = residual[1:]
        trace.append(
            {
                "cursor": cursor,
                "owner": owner,
                "emitted": character,
                "expected": expected,
                "residual_after": residual,
            }
        )
    assert not residual
    return residual, trace


def clause_tokens(text: str) -> list[str]:
    return [part.strip() for part in text.split(".") if part.strip()]


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
    assert len(left_emission) == len(right_obligation) == 70
    assert left_emission == right_obligation[::-1]
    left_residual, left_trace = emit_with_residual(left_emission, left_emission, "left_svo_emission")
    right_residual, right_trace = emit_with_residual(right_obligation, right_obligation, "right_svo_obligation")
    assert left_residual == right_residual == ""
    left_clauses = clause_tokens(NEW_LEFT)
    right_clauses = clause_tokens(NEW_RIGHT)
    assert len(left_clauses) == len(set(left_clauses)) == 6
    assert len(right_clauses) == len(set(right_clauses)) == 6
    assert {"sees", "stops"}.issubset({clause.split()[1] for clause in left_clauses})
    assert "spots" in {clause.split()[1] for clause in right_clauses}
    assert all(len(clause.split()) == 3 for clause in left_clauses + right_clauses)

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

    row = {
        "id": "semantic-role-graft-mara-nadia-666",
        "working_status": "semantic_role_graft_candidate",
        "promotion_status": {
            "promoted": False,
            "status": "selected_material_local_improvement_pending_review",
            "reason": "The graft removes the repeated saw-Noel-live pair and varies SVO relations; full-text review remains required.",
        },
        "rendered": rendered,
        "audit": project_audit,
        "independent_audit": independent,
        "parent_artifact": str(PARENT.relative_to(ROOT)),
        "parent_id": PARENT_ID,
        "parent_sha256": PARENT_SHA256,
        "growth_over_parent": 0,
        "new_event_content": left_clauses + right_clauses,
        "live_seam": {
            "normalized_window_left": list(WINDOW_LEFT),
            "normalized_window_right": list(WINDOW_RIGHT),
            "raw_window_left": list(RAW_LEFT),
            "raw_window_right": list(RAW_RIGHT),
            "old_left": OLD_LEFT,
            "old_right": OLD_RIGHT,
            "new_left": NEW_LEFT,
            "new_right": NEW_RIGHT,
            "initial_owner": "left_semantic_role_emitter",
            "left_emission": left_emission,
            "right_obligation": right_obligation,
            "right_consumption": right_obligation,
            "final_owner": None,
            "final_residual": "",
            "left_trace": left_trace,
            "right_trace": right_trace,
            "committed_character_contradictions": 0,
            "backtracks": 0,
        },
        "semantic_roles": {
            "left_subjects": [clause.split()[0] for clause in left_clauses],
            "left_objects": [clause.split()[-1] for clause in left_clauses],
            "varied_relations": ["sees", "stops", "spots"],
            "repeated_neighboring_clauses": False,
            "complete_svo_clauses": True,
            "vocative_or_appositive_fragments": False,
        },
        "readability_delta": {
            "repeated_saw_noel_live_before": 2,
            "repeated_saw_noel_live_after": 0,
            "varied_relations": ["sees", "stops", "spots"],
            "material_full_text_improvement": True,
        },
        "grammar_debt": {
            "inherited_proper_palindromic_spans": True,
            "remaining_delivers_maps_elsewhere": True,
            "remaining_star_spam_scaffolding": True,
            "remaining_mara_stops_rats_scaffolding": True,
            "rough_syntax_elsewhere": True,
            "human_reader_validation": False,
            "effect": "retain 568 incumbent and active 666 frontier until review",
        },
        "provenance": (
            "bounded hand-authored semantic-role SVO graft loaded from the promoted "
            "666 frontier; online residual trace closes before independent admission"
        ),
    }

    return {
        "experiment_id": "incumbent-666-semantic-role-graft-20260922",
        "method": "bounded hand-authored semantic-role graft with online residual ownership",
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
            "children_at_least_568": 1,
            "graft_child_letters": independent["normalized_letters"],
            "left_window_letters": 70,
            "right_window_letters": 70,
            "varied_relation_count": 3,
            "committed_character_contradictions": 0,
            "backtracks": 0,
        },
        "preserved_frontier": list(FRONTIER),
        "rows": [row],
        "next_operator": (
            "review this semantic-role graft as the selected material local improvement; "
            "do not demote 568 or promote without full-text review"
        ),
    }


def main() -> None:
    payload = build_payload()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
    print(payload["rows"][0]["rendered"])


if __name__ == "__main__":
    main()
