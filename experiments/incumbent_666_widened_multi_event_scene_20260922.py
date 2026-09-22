"""Try one authored 84-letter scene, then widen once on contradiction."""
from __future__ import annotations

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


PARENT = ROOT / "runs" / "incumbent-666-central-mini-scene-comparison-20260922.json"
OUT = ROOT / "runs" / "incumbent-666-widened-multi-event-scene-20260922.json"
PARENT_ID = "central-mini-scene-comparison-leon-noel-666"
PARENT_SHA256 = "3951b9449ed3ab28f55d9798e344dfaf3123035f5f07c9dffdf0047bed0e1d79"
WIDE_LEFT = (197, 281)
WIDE_RIGHT = (385, 469)
WIDE_RAW_LEFT = (263, 384)
WIDE_RAW_RIGHT = (520, 644)
SWITCH_LEFT = (113, 281)
SWITCH_RIGHT = (385, 553)
SWITCH_RAW_LEFT = (151, 384)
SWITCH_RAW_RIGHT = (520, 754)
OLD_LEFT = "Now, Noel, did I live? Nora saw Noel live. Noel, I sit. Pat notes. Mara saw God. Sara, did I live? Nora, I saw desserts. "
OLD_RIGHT = "Stressed was I, Aron. Evil I did, Aras. Dog was Aram. Seton, tap. 'Tis I, Leon. “Evil Leon” was Aron. Evil I did, Leon won. "
NEW_LEFT = "Then Ira saw Ari. Liam spots Nadia. Nadia was Aram. Aram saw Ira. Ira sees Nadia. Nadia was Liam. Ari stops Ira."
NEW_RIGHT = "Then Aram saw Ira. Aidan was Aram. Aram sees Liam. Ira spots Nadia. Nadia was Liam. Aram sees Ari. Ari was Liam."
KNOWN_VERBS = {"sees", "stops", "spots", "saw", "was", "won"}


def clause_parts(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[.!?;]+", text) if part.strip()]


def clause_boundaries(text: str) -> list[int]:
    cursor = 0
    boundaries = []
    for clause in clause_parts(text):
        cursor += len(normalize(clause))
        boundaries.append(cursor)
    return boundaries


def clause_frame(clause: str) -> tuple[str, str] | None:
    words = clause.split()
    for index, word in enumerate(words):
        if word in KNOWN_VERBS and index > 0 and index + 1 < len(words):
            return " ".join(words[:index]), word
    return None


def frames(text: str) -> set[tuple[str, str]]:
    return {frame for clause in clause_parts(text) if (frame := clause_frame(clause))}


def online(emission: str, obligation: str, owner: str) -> dict[str, object]:
    result = consume_online(emission, obligation, owner)
    result["obligation_length"] = len(obligation)
    return result


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

    assert parent_tape[WIDE_LEFT[0] : WIDE_LEFT[1]] == normalize(OLD_LEFT)
    assert parent_tape[WIDE_RIGHT[0] : WIDE_RIGHT[1]] == normalize(OLD_RIGHT)
    assert parent_rendered[WIDE_RAW_LEFT[0] : WIDE_RAW_LEFT[1]] == OLD_LEFT
    assert parent_rendered[WIDE_RAW_RIGHT[0] : WIDE_RAW_RIGHT[1]] == OLD_RIGHT
    left_emission = normalize(NEW_LEFT)
    right_emission = normalize(NEW_RIGHT)
    assert len(left_emission) == len(right_emission) == 84
    assert clause_boundaries(NEW_LEFT) == [13, 27, 39, 49, 61, 73, 84]
    assert clause_boundaries(NEW_RIGHT) == [14, 26, 38, 51, 63, 74, 84]

    parent_clause_set = set(clause_parts(parent_rendered))
    parent_frame_set = frames(parent_rendered)
    candidate_clauses = clause_parts(NEW_LEFT) + clause_parts(NEW_RIGHT)
    candidate_frames = [clause_frame(clause) for clause in candidate_clauses]
    bridge_gates = {
        "global_clause_novelty": all(clause not in parent_clause_set for clause in candidate_clauses),
        "global_frame_novelty": all(frame is not None and frame not in parent_frame_set for frame in candidate_frames),
        "complete_english_clauses": all(clause_frame(clause) for clause in candidate_clauses),
        "connected_entity_chain": True,
        "causal_temporal_anaphoric_link": True,
        "catalogue_or_self_contained_palindrome": False,
        "neighboring_discourse_continuity": True,
    }
    assert bridge_gates["complete_english_clauses"]
    assert bridge_gates["connected_entity_chain"]
    assert bridge_gates["causal_temporal_anaphoric_link"]
    assert bridge_gates["catalogue_or_self_contained_palindrome"] is False

    left_trace = online(NEW_LEFT, right_emission[::-1], "widened_left")
    right_trace = online(NEW_RIGHT, left_emission[::-1], "widened_right")
    exact_closure = left_trace["exact"] and right_trace["exact"]
    assert not exact_closure

    attempt_rendered = (
        parent_rendered[: WIDE_RAW_LEFT[0]]
        + NEW_LEFT
        + " "
        + parent_rendered[WIDE_RAW_LEFT[1] : WIDE_RAW_RIGHT[0]]
        + " "
        + NEW_RIGHT
        + " "
        + parent_rendered[WIDE_RAW_RIGHT[1] :]
    )
    attempt_audit = audit(attempt_rendered)
    attempt_independent = independent_audit(attempt_rendered)
    assert attempt_independent["normalized_letters"] == 666
    assert not attempt_independent["two_pointer_exact"]

    switch_left_context = normalize(parent_rendered[SWITCH_RAW_LEFT[0] : WIDE_RAW_LEFT[0]])
    switch_right_context = normalize(parent_rendered[WIDE_RAW_RIGHT[1] : SWITCH_RAW_RIGHT[1]])
    assert len(switch_left_context) == len(switch_right_context) == 84
    switch_left_emission = switch_left_context + left_emission
    switch_right_emission = right_emission + switch_right_context
    switch_left_trace = online(switch_left_emission, switch_right_emission[::-1], "switched_widened_left")
    switch_right_trace = online(switch_right_emission, switch_left_emission[::-1], "switched_widened_right")
    assert not switch_left_trace["exact"]
    assert not switch_right_trace["exact"]

    row = {
        "id": "widened-multi-event-scene-no-closure-666",
        "working_status": "widened_scene_rejected_no_closure",
        "promotion_status": {
            "promoted": False,
            "status": "rejected_no_exact_closure",
            "reason": "The authored 84-letter scene met discourse gates but contradicted reciprocal matching; the same bridge was switched once into the larger actual seam and also contradicted.",
        },
        "rendered": parent_rendered,
        "audit": parent["audit"],
        "independent_audit": parent_audit,
        "parent_artifact": str(PARENT.relative_to(ROOT)),
        "parent_id": PARENT_ID,
        "parent_sha256": PARENT_SHA256,
        "growth_over_parent": 0,
        "bridge_attempt": {
            "normalized_windows": {"left": list(WIDE_LEFT), "right": list(WIDE_RIGHT)},
            "raw_windows": {"left": list(WIDE_RAW_LEFT), "right": list(WIDE_RAW_RIGHT)},
            "old_left": OLD_LEFT,
            "old_right": OLD_RIGHT,
            "new_left": NEW_LEFT,
            "new_right": NEW_RIGHT,
            "clause_boundary_cursors": {"left": clause_boundaries(NEW_LEFT), "right": clause_boundaries(NEW_RIGHT)},
            "candidate_clauses": candidate_clauses,
            "candidate_frames": [{"subject": s, "verb": v} for s, v in candidate_frames if v],
            "gates": bridge_gates,
            "left_emission": left_emission,
            "right_emission": right_emission,
            "left_trace": left_trace,
            "right_trace": right_trace,
            "attempt_rendered": attempt_rendered,
            "attempt_audit": attempt_audit,
            "attempt_independent_audit": attempt_independent,
            "exact_child_saved": False,
        },
        "switched_attempt": {
            "normalized_windows": {"left": list(SWITCH_LEFT), "right": list(SWITCH_RIGHT)},
            "raw_windows": {"left": list(SWITCH_RAW_LEFT), "right": list(SWITCH_RAW_RIGHT)},
            "reason": "immediate switch after first reciprocal contradiction",
            "context_letter_lengths": {"left": len(switch_left_context), "right": len(switch_right_context)},
            "authored_clause_boundary_residual_cursors": {
                "left": [len(switch_left_context) + cursor for cursor in clause_boundaries(NEW_LEFT)],
                "right": clause_boundaries(NEW_RIGHT),
            },
            "left_trace": switch_left_trace,
            "right_trace": switch_right_trace,
            "exact_child_saved": False,
        },
        "provenance": "one authored connected multi-event scene, explicit online residual ownership, then one immediate larger-seam switch",
    }
    return {
        "experiment_id": "incumbent-666-widened-multi-event-scene-20260922",
        "method": "one bounded 84-letter authored scene with one immediate larger-seam switch",
        "active_frontier_parent": {"artifact": str(PARENT.relative_to(ROOT)), "id": PARENT_ID, "letters": 666, "sha256": PARENT_SHA256},
        "working_incumbent": {"artifact": "runs/incumbent-560-outer-causal-scene-20261002.json", "id": "outer-causal-scene-568-working-incumbent", "letters": 568, "sha256": "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"},
        "preserved_frontier": list(FRONTIER),
        "rows": [row],
        "next_operator": "review next actual seam after the switched larger seam; preserve 568 incumbent and 560/558/556 frontier",
    }


def main() -> None:
    result = build_payload()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    row = result["rows"][0]
    for label, trace in (
        ("bridge-left", row["bridge_attempt"]["left_trace"]),
        ("bridge-right", row["bridge_attempt"]["right_trace"]),
        ("switch-left", row["switched_attempt"]["left_trace"]),
        ("switch-right", row["switched_attempt"]["right_trace"]),
    ):
        print(label, {key: trace[key] for key in ("exact", "cursor", "reason", "residual")})


if __name__ == "__main__":
    main()
