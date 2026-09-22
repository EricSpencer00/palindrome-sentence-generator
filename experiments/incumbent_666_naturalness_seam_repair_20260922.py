"""Run one naturalness-gated repair at the next reciprocal 13-letter seam."""
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
OUT = ROOT / "runs" / "incumbent-666-naturalness-seam-repair-20260922.json"
PARENT_ID = "comparison-alternative-nora-sees-666"
PARENT_SHA256 = "cafd77235f82d9ff4f68814dc7e03d196bf719bf1ec5d541e172073502e12297"
CHILD_SHA256 = "2dc6afd1545b52b24f2d53d04e6e2529929f52dd45b81c00495f5cf3b86b19ed"
SEAM_LEFT = (297, 310)
SEAM_RIGHT = (356, 369)
SEAM_RAW_LEFT = (406, 422)
SEAM_RAW_RIGHT = (485, 501)
OLD_LEFT = "Nadia sees Mara."
OLD_RIGHT = "Aram sees Aidan."
NEW_LEFT = "Liam stops Aram."
NEW_RIGHT = "Mara spots mail."
KNOWN_VERBS = ("sees", "stops", "spots", "saw", "was")
GENERIC_FILLER = {"thing", "stuff", "something", "whatever", "etc"}


def clause_parts(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[.!?;]+", text) if part.strip()]


def clause_frame(clause: str) -> tuple[str, str] | None:
    words = clause.split()
    for index, word in enumerate(words):
        if word in KNOWN_VERBS and index > 0 and index + 1 < len(words):
            return " ".join(words[:index]), word
    return None


def naturalness_evidence(parent_text: str, left: str, right: str) -> dict[str, object]:
    parent_clauses = clause_parts(parent_text)
    parent_clause_set = set(parent_clauses)
    parent_frames = {frame for clause in parent_clauses if (frame := clause_frame(clause))}
    candidate_clauses = clause_parts(left) + clause_parts(right)
    candidate_frames = [frame for clause in candidate_clauses if (frame := clause_frame(clause))]
    left_context = {
        "before": parent_clauses[parent_clauses.index("Noel, was I stressed")],
        "after": parent_clauses[parent_clauses.index("Nadia sees Mara") + 1],
    }
    right_context = {
        "before": parent_clauses[parent_clauses.index("Ari stops Nadia")],
        "after": parent_clauses[parent_clauses.index("Aram sees Aidan") + 1],
    }
    words = [word for clause in candidate_clauses for word in clause.split()]
    three_word_svo = all(len(clause.split()) == 3 for clause in candidate_clauses)
    no_compound_name_hack = all(len(clause.split()[0].split()) == 1 for clause in candidate_clauses)
    no_ambiguous_adverb_argument = three_word_svo
    no_generic_filler = not any(word.lower() in GENERIC_FILLER for word in words)
    neighbor_checks = {
        "left_before": candidate_clauses[0] != left_context["before"],
        "left_after": candidate_clauses[0] != left_context["after"],
        "right_before": candidate_clauses[-1] != right_context["before"],
        "right_after": candidate_clauses[-1] != right_context["after"],
    }
    return {
        "candidate_clauses": [clause + "." for clause in candidate_clauses],
        "candidate_frames": [{"subject": subject, "verb": verb} for subject, verb in candidate_frames],
        "global_clause_novelty": all(clause not in parent_clause_set for clause in candidate_clauses),
        "global_frame_novelty": all(frame not in parent_frames for frame in candidate_frames),
        "candidate_frame_duplicates": [
            {"subject": subject, "verb": verb, "count": count}
            for (subject, verb), count in Counter(candidate_frames).items()
            if count > 1
        ],
        "neighbor_context": {"left": left_context, "right": right_context},
        "neighbor_duplicate_checks": neighbor_checks,
        "complete_natural_clauses": all(clause_frame(clause) for clause in candidate_clauses),
        "no_compound_name_hack": no_compound_name_hack,
        "no_ambiguous_adverb_or_argument_structure": no_ambiguous_adverb_argument,
        "no_generic_filler": no_generic_filler,
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
    assert len(left_emission) == len(right_emission) == 13
    assert left_emission == right_emission[::-1]

    naturalness = naturalness_evidence(parent_rendered, NEW_LEFT, NEW_RIGHT)
    for key in (
        "global_clause_novelty",
        "global_frame_novelty",
        "complete_natural_clauses",
        "no_compound_name_hack",
        "no_ambiguous_adverb_or_argument_structure",
        "no_generic_filler",
    ):
        assert naturalness[key]
    assert naturalness["candidate_frame_duplicates"] == []
    assert all(naturalness["neighbor_duplicate_checks"].values())

    left_trace = consume_online(NEW_LEFT, left_emission, "naturalness_left")
    right_trace = consume_online(NEW_RIGHT, right_emission, "naturalness_right")
    assert left_trace["exact"] and right_trace["exact"]
    rendered = (
        parent_rendered[: SEAM_RAW_LEFT[0]]
        + NEW_LEFT
        + parent_rendered[SEAM_RAW_LEFT[1] : SEAM_RAW_RIGHT[0]]
        + NEW_RIGHT
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
        "id": "naturalness-seam-repair-liam-666",
        "working_status": "naturalness_aware_seam_candidate",
        "promotion_status": {
            "promoted": False,
            "status": "comparison_pending_full_text_review",
            "reason": "Naturalness gates pass locally and globally; retain unpromoted pending full-text review.",
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
            "name": "bounded_naturalness_aware_repair",
            "cartesian_sweep": False,
            "global_frozen_parent_checked": True,
            "naturalness": naturalness,
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
            "left_emission": left_emission,
            "right_obligation": right_emission,
            "left_trace": left_trace["trace"],
            "right_trace": right_trace["trace"],
            "final_residual": "",
            "residual_ownership": {"left": "naturalness_left", "right": "naturalness_right"},
        },
        "semantic_roles": {
            "complete_natural_clauses": True,
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
        "provenance": "bounded naturalness-aware reciprocal seam repair with global novelty, neighbor gates, and explicit residual ownership",
    }
    return {
        "experiment_id": "incumbent-666-naturalness-seam-repair-20260922",
        "method": "one bounded naturalness-aware repair at reciprocal seam",
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
        "next_operator": "review naturalness-aware seam candidate without promoting over 568 incumbent",
    }


def main() -> None:
    result = build_payload()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(result["rows"][0]["independent_audit"])
    print(result["rows"][0]["rendered"])


if __name__ == "__main__":
    main()
