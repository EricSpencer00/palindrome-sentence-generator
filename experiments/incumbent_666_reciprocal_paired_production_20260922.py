"""Atomically produce one reciprocal two-sided clause pair on the 666 parent."""
from __future__ import annotations

import json
import re
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


PARENT = ROOT / "runs" / "incumbent-666-central-mini-scene-comparison-20260922.json"
OUT = ROOT / "runs" / "incumbent-666-reciprocal-paired-production-20260922.json"
PARENT_ID = "central-mini-scene-comparison-leon-noel-666"
PARENT_SHA256 = "3951b9449ed3ab28f55d9798e344dfaf3123035f5f07c9dffdf0047bed0e1d79"
LEFT_WINDOW = (140, 156)
RIGHT_WINDOW = (510, 526)
LEFT_RAW = (188, 209)
RIGHT_RAW = (698, 719)
OLD_LEFT = "Nadia saw Noel live. "
OLD_RIGHT = "Evil Leon was Aidan. "
NEW_LEFT = "Evil Mara was Nadia. "
NEW_RIGHT = "Aidan saw Aram live. "
MAX_PAIRED_EXPANSIONS = 8


def compare_stream(emission: str, obligation: str, owner: str) -> dict[str, object]:
    emitted = normalize(emission)
    trace: list[dict[str, object]] = []
    for cursor, character in enumerate(emitted):
        expected = obligation[cursor] if cursor < len(obligation) else None
        item = {
            "cursor": cursor,
            "owner": owner,
            "emitted": character,
            "expected": expected,
            "residual_before": obligation[cursor:],
        }
        if expected != character:
            item["reason"] = "character_contradiction"
            trace.append(item)
            return {
                "exact": False,
                "cursor": cursor,
                "residual": obligation[cursor:],
                "reason": "character_contradiction",
                "owner": owner,
                "trace": trace,
            }
        item["reason"] = "matched"
        trace.append(item)
    residual = obligation[len(emitted) :]
    return {
        "exact": not residual,
        "cursor": len(emitted),
        "residual": residual,
        "reason": "closed" if not residual else "nonempty_residual",
        "owner": owner,
        "trace": trace,
    }


FRAME_VERBS = {"was", "saw", "sees", "stops", "spots", "won", "did", "notes", "live", "tap", "sit"}


def extract_frames(rendered: str) -> set[str]:
    """Extract subject|verb frames from raw clause tokens before normalization."""
    frames: set[str] = set()
    for raw_clause in re.split(r"[.!?;]+", rendered):
        words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", raw_clause)
        for index, word in enumerate(words):
            verb = word.lower()
            if verb in FRAME_VERBS and index:
                subject = " ".join(words[:index]).lower()
                frames.add(f"{subject}|{verb}")
                break
    return frames


def boundary_context(rendered: str, raw_window: tuple[int, int]) -> dict[str, object]:
    before = rendered[max(0, raw_window[0] - 48) : raw_window[0]]
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
    parent_independent = independent_audit(parent_rendered)
    assert parent_independent["normalized_letters"] == 666
    assert parent_independent["two_pointer_exact"]
    assert parent_independent["sha256_forward"] == PARENT_SHA256
    assert parent["promotion_status"]["promoted"] is True
    for entry in FRONTIER:
        validate_frontier_entry(entry)

    assert parent_tape[LEFT_WINDOW[0] : LEFT_WINDOW[1]] == normalize(OLD_LEFT)
    assert parent_tape[RIGHT_WINDOW[0] : RIGHT_WINDOW[1]] == normalize(OLD_RIGHT)
    assert parent_rendered[LEFT_RAW[0] : LEFT_RAW[1]] == OLD_LEFT
    assert parent_rendered[RIGHT_RAW[0] : RIGHT_RAW[1]] == OLD_RIGHT

    left_emission = normalize(NEW_LEFT)
    right_emission = normalize(NEW_RIGHT)
    assert len(left_emission) == len(right_emission) == 16
    assert left_emission == right_emission[::-1]

    # This paired-production operator authors a new reciprocal equation at
    # the live seam; each side's obligation is derived from the other newly
    # emitted side, not from the superseded parent shell.
    left_obligation = right_emission[::-1]
    right_obligation = left_emission[::-1]
    left_stream = compare_stream(NEW_LEFT, left_obligation, "left")
    right_stream = compare_stream(NEW_RIGHT, right_obligation, "right")
    assert left_stream["exact"] and right_stream["exact"]

    parent_clauses = {clause.strip().lower() for clause in re.findall(r"[^.!?;]+", parent_rendered)}
    assert NEW_LEFT.strip().lower() not in parent_clauses
    assert NEW_RIGHT.strip().lower() not in parent_clauses
    parent_frames = extract_frames(parent_rendered)
    candidate_frames = extract_frames(NEW_LEFT) | extract_frames(NEW_RIGHT)
    assert candidate_frames == {"evil mara|was", "aidan|saw"}
    assert candidate_frames.isdisjoint(parent_frames)

    paired_state_before = {
        "left_cursor": 0,
        "right_cursor": 0,
        "left_residual": left_obligation,
        "right_residual": right_obligation,
        "left_owner": "left",
        "right_owner": "right",
        "left_subject_object_stack": {"subjects": ["Nadia"], "objects": ["Noel"]},
        "right_subject_object_stack": {"subjects": ["Aidan"], "objects": ["Aram"]},
        "left_active_discourse_entity": "Nadia",
        "right_active_discourse_entity": "Aram",
    }
    paired_state_after = {
        "left_cursor": 16,
        "right_cursor": 16,
        "left_residual": left_stream["residual"],
        "right_residual": right_stream["residual"],
        "left_owner": "left",
        "right_owner": "right",
        "left_subject_object_stack": {"subject": "Evil Mara", "object": "Nadia"},
        "right_subject_object_stack": {"subject": "Aidan", "object": "Aram"},
        "left_active_discourse_entity": "Nadia",
        "right_active_discourse_entity": "Aram",
        "atomic_entity_update": {
            "left": {"from": "Mara", "to": "Nadia"},
            "right": {"from": "Aidan", "to": "Aram"},
        },
    }
    paired_expansion = {
        "expansion_index": 1,
        "left_emission": NEW_LEFT,
        "right_emission": NEW_RIGHT,
        "required_left_prefix": left_obligation,
        "required_right_prefix": right_obligation,
        "paired_cursors_before": [0, 0],
        "paired_cursors_after": [16, 16],
        "owners": ["left", "right"],
        "state_before": paired_state_before,
        "left_stream": left_stream,
        "right_stream": right_stream,
        "atomic_state_update": paired_state_after,
        "complete_finite_clause": {"left": True, "right": True},
        "global_clause_novelty": {"left": True, "right": True},
        "global_frame_novelty": {"left": True, "right": True},
        "frame_novelty_evidence": {
            "parent_frames_extracted_from_rendered_tokens": sorted(parent_frames),
            "candidate_frames_extracted_from_rendered_tokens": sorted(candidate_frames),
            "candidate_frames_absent_from_parent": True,
        },
        "boundary_context": {
            "left": boundary_context(parent_rendered, LEFT_RAW),
            "right": boundary_context(parent_rendered, RIGHT_RAW),
        },
        "fragment_or_catalogue_rejection": False,
        "accepted": True,
    }
    assert paired_expansion["paired_cursors_after"] == [16, 16]
    assert paired_state_after["left_residual"] == ""
    assert paired_state_after["right_residual"] == ""

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
    assert child_independent["sha_equal"]
    assert child_audit["project_validator_exact"]

    row = {
        "id": "reciprocal-paired-production-mara-nadia-666",
        "working_status": "comparison_frontier_alternative",
        "promotion_status": {
            "promoted": False,
            "status": "pending_full_text_readability_review",
            "reason": "The paired production is independently exact and globally novel on both sides, but full-text semantic and repetition debt remains; it stays unpromoted pending readability review.",
        },
        "rendered": child_rendered,
        "audit": child_audit,
        "independent_audit": child_independent,
        "parent_artifact": str(PARENT.relative_to(ROOT)),
        "parent_id": PARENT_ID,
        "parent_sha256": PARENT_SHA256,
        "growth_over_parent": 0,
        "new_event_content": ["Evil Mara was Nadia", "Aidan saw Aram live"],
        "full_text_review": {
            "material_full_text_improvement": False,
            "semantic_debt": [
                "Evil Mara was Nadia is an identity-style clause whose event meaning is ambiguous.",
                "Aidan saw Aram live has a bare live adjunct and remains telegraphic in context.",
            ],
            "repetition_debt_details": [
                "The replacement changes only one paired seam and inherits the surrounding repeated stops/spots scaffold.",
                "The child does not yet demonstrate a material full-text readability improvement over the promoted parent.",
            ],
            "grammar_debt": True,
            "repetition_debt_present": True,
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
            "paired_expansions": [paired_expansion],
            "paired_expansion_count": 1,
            "max_paired_expansions": MAX_PAIRED_EXPANSIONS,
            "paired_cursors_after": [16, 16],
            "residuals": {"left": "", "right": ""},
            "clause_boundary_cursors": {"left": [16], "right": [16]},
            "exact_reverse_equation": True,
        },
        "switch_after_rejection": {"attempted": False, "reason": "primary paired production accepted"},
        "provenance": "one atomic reciprocal paired production from the promoted 666 parent",
    }
    return {
        "experiment_id": "incumbent-666-reciprocal-paired-production-20260922",
        "method": "atomic reciprocal paired production; one seam; at most eight paired expansions",
        "active_frontier_parent": {"artifact": str(PARENT.relative_to(ROOT)), "id": PARENT_ID, "letters": 666, "sha256": PARENT_SHA256},
        "working_incumbent": {"artifact": "runs/incumbent-560-outer-causal-scene-20261002.json", "id": "outer-causal-scene-568-working-incumbent", "letters": 568, "sha256": "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"},
        "preserved_frontier": list(FRONTIER),
        "rows": [row],
        "next_operator": "full-text review of paired 666 comparison; preserve 568 incumbent and 560/558/556 frontier",
    }


def main() -> None:
    result = build_payload()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    row = result["rows"][0]
    print({
        "id": row["id"],
        "letters": row["independent_audit"]["normalized_letters"],
        "sha256": row["independent_audit"]["sha256_forward"],
        "paired_cursors": row["live_seam"]["paired_expansions"][0]["paired_cursors_after"],
        "residuals": row["live_seam"]["residuals"],
    })


if __name__ == "__main__":
    main()
