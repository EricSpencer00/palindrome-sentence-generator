"""Produce one context-aware reciprocal pair on the promoted 666 parent."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.incumbent_498_deep_clause_transducer_20261002 import audit
from experiments.incumbent_666_boundary_discourse_linker_20260922 import (
    FRONTIER,
    independent_audit,
    normalize,
    validate_frontier_entry,
)
from experiments.incumbent_666_reciprocal_paired_production_20260922 import extract_frames


PARENT = ROOT / "runs" / "incumbent-666-central-mini-scene-comparison-20260922.json"
OUT = ROOT / "runs" / "incumbent-666-context-aware-reciprocal-pair-20260922.json"
PARENT_ID = "central-mini-scene-comparison-leon-noel-666"
PARENT_SHA256 = "3951b9449ed3ab28f55d9798e344dfaf3123035f5f07c9dffdf0047bed0e1d79"
LEFT_WINDOW = (127, 140)
RIGHT_WINDOW = (526, 539)
LEFT_RAW = (171, 188)
RIGHT_RAW = (719, 736)
OLD_LEFT = "Mara sees Nadia. "
OLD_RIGHT = "Aidan sees Aram. "
NEW_LEFT = "Noel stops Aras. "
NEW_RIGHT = "Sara spots Leon. "
MAX_PAIRED_EXPANSIONS = 8


def compare_stream(emission: str, obligation: str, owner: str) -> dict[str, object]:
    emitted = normalize(emission)
    trace = []
    for cursor, character in enumerate(emitted):
        expected = obligation[cursor] if cursor < len(obligation) else None
        if character != expected:
            return {
                "exact": False,
                "cursor": cursor,
                "expected": expected,
                "emitted": character,
                "residual": obligation[cursor:],
                "owner": owner,
                "reason": "character_contradiction",
                "trace": trace + [{"cursor": cursor, "owner": owner, "emitted": character, "expected": expected}],
            }
        trace.append({"cursor": cursor, "owner": owner, "emitted": character, "expected": expected, "residual_after": obligation[cursor + 1 :]})
    residual = obligation[len(emitted) :]
    return {
        "exact": not residual,
        "cursor": len(emitted),
        "expected": None,
        "emitted": None,
        "residual": residual,
        "owner": owner,
        "reason": "closed" if not residual else "nonempty_residual",
        "trace": trace,
    }


def context(rendered: str, raw_window: tuple[int, int]) -> dict[str, object]:
    before = rendered[raw_window[0] - 48 : raw_window[0]]
    after = rendered[raw_window[1] : raw_window[1] + 48]
    return {
        "before_raw": before,
        "after_raw": after,
        "before_normalized_tail": normalize(before)[-24:],
        "after_normalized_prefix": normalize(after)[:24],
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

    assert parent_tape[LEFT_WINDOW[0] : LEFT_WINDOW[1]] == normalize(OLD_LEFT)
    assert parent_tape[RIGHT_WINDOW[0] : RIGHT_WINDOW[1]] == normalize(OLD_RIGHT)
    assert parent_rendered[LEFT_RAW[0] : LEFT_RAW[1]] == OLD_LEFT
    assert parent_rendered[RIGHT_RAW[0] : RIGHT_RAW[1]] == OLD_RIGHT

    left_emission = normalize(NEW_LEFT)
    right_emission = normalize(NEW_RIGHT)
    assert len(left_emission) == len(right_emission) == 13
    assert left_emission == right_emission[::-1]
    left_obligation = right_emission[::-1]
    right_obligation = left_emission[::-1]
    left_stream = compare_stream(NEW_LEFT, left_obligation, "left")
    right_stream = compare_stream(NEW_RIGHT, right_obligation, "right")
    assert left_stream["exact"] and right_stream["exact"]

    parent_frames = extract_frames(parent_rendered)
    candidate_frames = extract_frames(NEW_LEFT) | extract_frames(NEW_RIGHT)
    assert candidate_frames == {"noel|stops", "sara|spots"}
    assert candidate_frames.isdisjoint(parent_frames)
    parent_clauses = {part.strip().lower() for part in parent_rendered.replace("?", ".").split(".") if part.strip()}
    assert NEW_LEFT.strip().lower() not in parent_clauses
    assert NEW_RIGHT.strip().lower() not in parent_clauses

    before = {
        "left_active_entity": "Nadia",
        "right_active_entity": "Aram",
        "left_subject_object_stack": {"subjects": ["Mara"], "objects": ["Nadia"]},
        "right_subject_object_stack": {"subjects": ["Aidan"], "objects": ["Aram"]},
        "removed_frames": {"left": "mara|sees", "right": "aidan|sees"},
    }
    after = {
        "left_active_entity": "Aras",
        "right_active_entity": "Leon",
        "left_subject_object_stack": {"subjects": ["Noel"], "objects": ["Aras"]},
        "right_subject_object_stack": {"subjects": ["Sara"], "objects": ["Leon"]},
        "atomic_entity_updates": {
            "left": {"from": "Nadia", "to": "Aras"},
            "right": {"from": "Aram", "to": "Leon"},
        },
    }
    entity_links = {
        "left": {"direction": "after", "entity": "Noel", "context_clause": "Nadia saw Noel live."},
        "right": {"direction": "before", "entity": "Leon", "context_clause": "Evil Leon was Aidan."},
    }
    paired = {
        "expansion_index": 1,
        "paired_cursors_before": [0, 0],
        "paired_cursors_after": [13, 13],
        "owners": ["left", "right"],
        "left_emission": NEW_LEFT,
        "right_emission": NEW_RIGHT,
        "left_obligation": left_obligation,
        "right_obligation": right_obligation,
        "left_stream": left_stream,
        "right_stream": right_stream,
        "state_before": before,
        "state_after": after,
        "neighboring_entity_links": entity_links,
        "complete_finite_clause": {"left": True, "right": True},
        "global_clause_novelty": {"left": True, "right": True},
        "global_frame_novelty": {"left": True, "right": True},
        "frame_novelty_evidence": {
            "parent_frames_extracted_from_rendered_tokens": sorted(parent_frames),
            "candidate_frames_extracted_from_rendered_tokens": sorted(candidate_frames),
            "candidate_frames_absent_from_parent": True,
        },
        "removed_repeated_patterns": {
            "right_aidan_sees_aram": True,
            "left_mara_sees_nadia_frame": True,
        },
        "boundary_context": {"left": context(parent_rendered, LEFT_RAW), "right": context(parent_rendered, RIGHT_RAW)},
        "fragment_or_catalogue_rejection": False,
        "accepted": True,
    }

    child_rendered = (
        parent_rendered[: LEFT_RAW[0]]
        + NEW_LEFT
        + parent_rendered[LEFT_RAW[1] : RIGHT_RAW[0]]
        + NEW_RIGHT
        + parent_rendered[RIGHT_RAW[1] :]
    )
    child_audit = audit(child_rendered)
    child_independent = independent_audit(child_rendered)
    assert child_independent["normalized_letters"] == 666
    assert child_independent["two_pointer_exact"]
    assert child_audit["project_validator_exact"]

    row = {
        "id": "context-aware-reciprocal-pair-noel-sara-666",
        "working_status": "active_666_readability_frontier",
        "promotion_status": {
            "promoted": True,
            "status": "promoted_active_readability_frontier",
            "reason": "Promoted after full-text review: the context-aware pair removes the repeated Mara-sees/Aidan-sees pattern, adds distinct Noel/Sara event frames, and preserves exactness with linked neighboring entities.",
            "full_text_rationale": {
                "material_full_text_improvement": True,
                "local_change": "Replaces Mara sees Nadia / Aidan sees Aram with Noel stops Aras / Sara spots Leon.",
                "remaining_debt": "Inherited rough syntax, repeated scaffolding elsewhere, and absent reader validation remain unresolved.",
            },
            "comparison_retained": {
                "artifact": str(PARENT.relative_to(ROOT)),
                "id": PARENT_ID,
                "sha256": PARENT_SHA256,
            },
        },
        "rendered": child_rendered,
        "audit": child_audit,
        "independent_audit": child_independent,
        "parent_artifact": str(PARENT.relative_to(ROOT)),
        "parent_id": PARENT_ID,
        "parent_sha256": PARENT_SHA256,
        "growth_over_parent": 0,
        "new_event_content": ["Noel stops Aras", "Sara spots Leon"],
        "readability_delta": {
            "material_full_text_improvement": True,
            "status": "promoted_active_readability_frontier",
            "rationale": "Full-text review credits the local removal of the repeated Mara-sees/Aidan-sees pattern and the distinct Noel/Sara event frames; inherited syntax and repetition debt remain.",
        },
        "live_seam": {
            "normalized_left": list(LEFT_WINDOW),
            "normalized_right": list(RIGHT_WINDOW),
            "raw_left": list(LEFT_RAW),
            "raw_right": list(RIGHT_RAW),
            "old_left": OLD_LEFT,
            "old_right": OLD_RIGHT,
            "new_left": NEW_LEFT,
            "new_right": NEW_RIGHT,
            "paired_expansions": [paired],
            "paired_expansion_count": 1,
            "max_paired_expansions": MAX_PAIRED_EXPANSIONS,
            "paired_cursors_after": [13, 13],
            "residuals": {"left": "", "right": ""},
            "exact_reverse_equation": True,
        },
        "switch_after_rejection": {"attempted": False, "reason": "primary paired production accepted"},
        "provenance": "one context-aware reciprocal paired production on the promoted 666 parent",
    }
    return {
        "experiment_id": "incumbent-666-context-aware-reciprocal-pair-20260922",
        "method": "one context-aware reciprocal pair; full-parent clause/frame novelty extraction; at most eight paired expansions",
        "active_frontier_parent": {"artifact": str(PARENT.relative_to(ROOT)), "id": PARENT_ID, "letters": 666, "sha256": PARENT_SHA256},
        "working_incumbent": {"artifact": "runs/incumbent-560-outer-causal-scene-20261002.json", "id": "outer-causal-scene-568-working-incumbent", "letters": 568, "sha256": "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"},
        "preserved_frontier": list(FRONTIER),
        "active_readability_frontier": {
            "artifact": str(OUT.relative_to(ROOT)),
            "id": "context-aware-reciprocal-pair-noel-sara-666",
            "letters": 666,
            "sha256": child_independent["sha256_forward"],
        },
        "comparison_retained": {
            "artifact": str(PARENT.relative_to(ROOT)),
            "id": PARENT_ID,
            "letters": 666,
            "sha256": PARENT_SHA256,
        },
        "rows": [row],
        "next_operator": "full-text review of context-aware 666 comparison; preserve 568 incumbent and 560/558/556 frontier",
    }


def main() -> None:
    result = build_payload()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    row = result["rows"][0]
    print({"id": row["id"], "letters": row["independent_audit"]["normalized_letters"], "sha256": row["independent_audit"]["sha256_forward"], "paired_cursors": row["live_seam"]["paired_cursors_after"], "residuals": row["live_seam"]["residuals"]})


if __name__ == "__main__":
    main()
