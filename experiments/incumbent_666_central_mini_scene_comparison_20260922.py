"""Build the reviewer-specified bounded central mini-scene comparison."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.incumbent_498_deep_clause_transducer_20261002 import audit
from experiments.incumbent_666_boundary_discourse_linker_20260922 import (
    FRONTIER,
    consume_online,
    independent_audit,
    normalize,
    validate_frontier_entry,
)


PARENT = ROOT / "runs" / "incumbent-666-comparison-alternative-20260922.json"
OUT = ROOT / "runs" / "incumbent-666-central-mini-scene-comparison-20260922.json"
PARENT_ID = "comparison-alternative-nora-sees-666"
PARENT_SHA256 = "cafd77235f82d9ff4f68814dc7e03d196bf719bf1ec5d541e172073502e12297"
CHILD_SHA256 = "3951b9449ed3ab28f55d9798e344dfaf3123035f5f07c9dffdf0047bed0e1d79"
SEAM_LEFT = (281, 323)
SEAM_RIGHT = (343, 385)
SEAM_RAW_LEFT = (384, 440)
SEAM_RAW_RIGHT = (468, 524)
OLD_LEFT = "Noel, was I stressed? Nadia sees Mara. Aidan spots Ira. "
OLD_RIGHT = "Ari stops Nadia. Aram sees Aidan. Desserts I saw, Leon. "
NEW_LEFT = "Leon stops Noel. Noel spots Nadia. Nadia stops Aidan."
NEW_RIGHT = "Nadia spots Aidan. Aidan stops Leon. Leon spots Noel."
KNOWN_VERBS = ("sees", "stops", "spots", "saw", "was", "won")


def clause_parts(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[.!?;]+", text) if part.strip()]


def clause_frame(clause: str) -> tuple[str, str] | None:
    words = clause.split()
    for index, word in enumerate(words):
        if word in KNOWN_VERBS and index > 0 and index + 1 < len(words):
            return " ".join(words[:index]), word
    return None


def novelty_evidence(parent_text: str) -> dict[str, object]:
    parent_clauses = clause_parts(parent_text)
    candidate_clauses = clause_parts(NEW_LEFT) + clause_parts(NEW_RIGHT)
    parent_frames = {frame for clause in parent_clauses if (frame := clause_frame(clause))}
    candidate_frames = [frame for clause in candidate_clauses if (frame := clause_frame(clause))]
    left_context = {
        "before": parent_clauses[parent_clauses.index("Nora, I saw desserts")],
        "after": parent_clauses[parent_clauses.index("Aidan spots Ira") + 1],
    }
    right_context = {
        "before": parent_clauses[parent_clauses.index("Ari stops Nadia") - 1],
        "after": parent_clauses[parent_clauses.index("Desserts I saw, Leon") + 1],
    }
    neighbor_checks = {
        "left_before": candidate_clauses[0] != left_context["before"],
        "left_after": candidate_clauses[2] != left_context["after"],
        "right_before": candidate_clauses[3] != right_context["before"],
        "right_after": candidate_clauses[-1] != right_context["after"],
    }
    return {
        "candidate_clauses": [clause + "." for clause in candidate_clauses],
        "candidate_frames": [{"subject": subject, "verb": verb} for subject, verb in candidate_frames],
        "global_clause_novelty": all(clause not in set(parent_clauses) for clause in candidate_clauses),
        "global_frame_novelty": all(frame not in parent_frames for frame in candidate_frames),
        "candidate_frame_duplicates": [
            {"subject": subject, "verb": verb, "count": count}
            for (subject, verb), count in Counter(candidate_frames).items()
            if count > 1
        ],
        "connected_entity_chain": {
            "left": ["Leon", "Noel", "Nadia", "Aidan"],
            "right": ["Nadia", "Aidan", "Leon", "Noel"],
            "shared_links": ["Leon", "Noel", "Nadia", "Aidan"],
        },
        "neighbor_context": {"left": left_context, "right": right_context},
        "neighbor_duplicate_checks": neighbor_checks,
        "complete_finite_svo_clauses": all(len(clause.split()) == 3 and clause_frame(clause) for clause in candidate_clauses),
        "cartesian_sweep": False,
    }


def build_payload() -> dict[str, object]:
    payload = json.loads(PARENT.read_text())
    parent = next(row for row in payload["rows"] if row["id"] == PARENT_ID)
    parent_rendered = str(parent["rendered"])
    parent_tape = normalize(parent_rendered)
    parent_audit = independent_audit(parent_rendered)
    assert parent_audit["normalized_letters"] == 666
    assert parent_audit["two_pointer_exact"]
    assert parent_audit["sha256_forward"] == PARENT_SHA256
    assert parent["promotion_status"]["promoted"] is True
    for entry in FRONTIER:
        validate_frontier_entry(entry)

    assert parent_tape[SEAM_LEFT[0] : SEAM_LEFT[1]] == normalize(OLD_LEFT)
    assert parent_tape[SEAM_RIGHT[0] : SEAM_RIGHT[1]] == normalize(OLD_RIGHT)
    assert parent_rendered[SEAM_RAW_LEFT[0] : SEAM_RAW_LEFT[1]] == OLD_LEFT
    assert parent_rendered[SEAM_RAW_RIGHT[0] : SEAM_RAW_RIGHT[1]] == OLD_RIGHT
    left_emission = normalize(NEW_LEFT)
    right_emission = normalize(NEW_RIGHT)
    assert len(left_emission) == len(right_emission) == 42
    assert left_emission == right_emission[::-1]
    assert [len(normalize(clause)) for clause in clause_parts(NEW_LEFT)] == [13, 14, 15]
    assert [len(normalize(clause)) for clause in clause_parts(NEW_LEFT)] == [13, 27 - 13, 42 - 27]

    novelty = novelty_evidence(parent_rendered)
    assert novelty["global_clause_novelty"]
    assert novelty["global_frame_novelty"]
    assert novelty["candidate_frame_duplicates"] == []
    assert novelty["connected_entity_chain"]["shared_links"] == ["Leon", "Noel", "Nadia", "Aidan"]
    assert all(novelty["neighbor_duplicate_checks"].values())
    assert novelty["complete_finite_svo_clauses"]

    left_trace = consume_online(NEW_LEFT, left_emission, "mini_scene_left")
    right_trace = consume_online(NEW_RIGHT, right_emission, "mini_scene_right")
    assert left_trace["exact"] and right_trace["exact"]
    rendered = (
        parent_rendered[: SEAM_RAW_LEFT[0]]
        + NEW_LEFT
        + " "
        + parent_rendered[SEAM_RAW_LEFT[1] : SEAM_RAW_RIGHT[0]]
        + NEW_RIGHT
        + " "
        + parent_rendered[SEAM_RAW_RIGHT[1] :]
    )
    project_audit = audit(rendered)
    independent = independent_audit(rendered)
    assert project_audit["letters"] == 666
    assert project_audit["project_validator_exact"]
    assert independent["normalized_letters"] == 666
    assert independent["two_pointer_exact"]
    assert independent["sha256_forward"] == CHILD_SHA256
    assert independent["sha_equal"]

    row = {
        "id": "central-mini-scene-comparison-leon-noel-666",
        "working_status": "central_mini_scene_comparison",
        "promotion_status": {
            "promoted": False,
            "status": "comparison_pending_full_text_review",
            "reason": "Reviewer-specified connected mini-scene closes exactly with global novelty; retain as comparison pending full-text review.",
        },
        "rendered": rendered,
        "audit": project_audit,
        "independent_audit": independent,
        "parent_artifact": str(PARENT.relative_to(ROOT)),
        "parent_id": PARENT_ID,
        "parent_sha256": PARENT_SHA256,
        "growth_over_parent": 0,
        "new_event_content": [NEW_LEFT, NEW_RIGHT],
        "operator": {
            "name": "bounded_connected_central_mini_scene",
            "cartesian_sweep": False,
            "global_frozen_parent_checked": True,
            "normalized_windows": {"left": list(SEAM_LEFT), "right": list(SEAM_RIGHT)},
            "raw_windows": {"left": list(SEAM_RAW_LEFT), "right": list(SEAM_RAW_RIGHT)},
            "novelty": novelty,
        },
        "live_seam": {
            "normalized_window_left": list(SEAM_LEFT),
            "normalized_window_right": list(SEAM_RIGHT),
            "raw_window_left": list(SEAM_RAW_LEFT),
            "raw_window_right": list(SEAM_RAW_RIGHT),
            "old_left": OLD_LEFT,
            "old_right": OLD_RIGHT,
            "new_left": NEW_LEFT,
            "new_right": NEW_RIGHT,
            "clause_boundary_cursors": [13, 27, 42],
            "left_emission": left_emission,
            "right_obligation": right_emission,
            "left_trace": left_trace["trace"],
            "right_trace": right_trace["trace"],
            "final_residual": "",
            "residual_ownership": {"left": "mini_scene_left", "right": "mini_scene_right"},
        },
        "semantic_roles": {
            "complete_finite_svo_clauses": True,
            "connected_entity_chains": True,
            "varied_relations": ["stops", "spots"],
            "repeated_subject_verb_frames": [],
            "repeated_neighboring_clauses": False,
        },
        "grammar_debt": {
            "inherited_proper_palindromic_spans": True,
            "remaining_full_text_repetition": True,
            "human_reader_validation": False,
            "effect": "retain 568 incumbent and promoted 666 parent until review",
        },
        "provenance": "reviewer-derived 42-letter connected mini-scene with exact clause boundaries and explicit residual ownership",
    }
    return {
        "experiment_id": "incumbent-666-central-mini-scene-comparison-20260922",
        "method": "one bounded connected central mini-scene comparison",
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
        "preserved_frontier": list(FRONTIER),
        "rows": [row],
        "next_operator": "review central mini-scene comparison without promoting over 568 incumbent",
    }


def main() -> None:
    result = build_payload()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(result["rows"][0]["independent_audit"])
    print(result["rows"][0]["rendered"])


if __name__ == "__main__":
    main()
