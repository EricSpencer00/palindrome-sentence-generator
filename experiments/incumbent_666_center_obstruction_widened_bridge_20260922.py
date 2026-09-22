"""Persist the 10-letter center obstruction, then attempt one widened bridge."""
from __future__ import annotations

import hashlib
import json
import re
import sys
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
OUT = ROOT / "runs" / "incumbent-666-center-obstruction-widened-bridge-20260922.json"
PARENT_ID = "central-mini-scene-comparison-leon-noel-666"
PARENT_SHA256 = "3951b9449ed3ab28f55d9798e344dfaf3123035f5f07c9dffdf0047bed0e1d79"
CENTER_LEFT = (323, 343)
CENTER_RAW = (438, 466)
WIDE_LEFT = (281, 333)
WIDE_RIGHT = (333, 385)
WIDE_RAW_LEFT = (384, 452)
WIDE_RAW_RIGHT = (452, 520)
NEXT_LEFT = (197, 281)
NEXT_RIGHT = (385, 469)
NEXT_RAW_LEFT = (263, 384)
NEXT_RAW_RIGHT = (520, 644)
CENTER_RENDERED = "Ari sees God. Dog sees Ira. "
OLD_LEFT = "Leon stops Noel. Noel spots Nadia. Nadia stops Aidan. Ari sees God. "
OLD_RIGHT = "Dog sees Ira. Nadia spots Aidan. Aidan stops Leon. Leon spots Noel. "
NEW_LEFT = "Aram spots Liam. Liam spots Nadia. Now Nadia was Aram, Aram saw Ira."
NEW_RIGHT = "Ira stops Aidan. Aidan stops Aram. Aram was Nadia. Nadia sees Aron."


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
        if word in {"sees", "stops", "spots", "saw", "was", "won"} and index > 0 and index + 1 < len(words):
            return " ".join(words[:index]), word
    return None


def parent_frames(text: str) -> set[tuple[str, str]]:
    return {frame for clause in clause_parts(text) if (frame := clause_frame(clause))}


def compare_online(emission: str, obligation: str, owner: str) -> dict[str, object]:
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

    center_tape = parent_tape[CENTER_LEFT[0] : CENTER_LEFT[1]]
    assert center_tape == normalize(CENTER_RENDERED)
    assert len(center_tape) == 20
    center_left_clause = normalize("Ari sees God.")
    center_right_clause = normalize("Dog sees Ira.")
    assert len(center_left_clause) == len(center_right_clause) == 10
    assert center_left_clause + center_right_clause == center_tape
    rejected_right = normalize("God sees Ari.")
    assert center_left_clause != rejected_right[::-1]
    center_obstruction = {
        "normalized_window": list(CENTER_LEFT),
        "raw_window": list(CENTER_RAW),
        "rendered": CENTER_RENDERED,
        "normalized_tape": center_tape,
        "length_pattern": {
            "clause_lengths": [10, 10],
            "token_letter_lengths": [[3, 4, 3], [3, 4, 3]],
            "split_cursor": 10,
            "available_total": 20,
        },
        "rejected_disconnected_pair": {
            "left": "Ari sees God.",
            "right": "God sees Ari.",
            "exact_reverse": False,
            "reason": "Although it is a 10-letter disconnected God/Ari alternative, it contradicts the required reciprocal tape (the exact reverse is Dog sees Ira) and does not continue the neighboring Aidan/Nadia discourse chain.",
        },
        "status": "obstruction_persisted",
    }

    assert parent_tape[WIDE_LEFT[0] : WIDE_LEFT[1]] == normalize(OLD_LEFT)
    assert parent_tape[WIDE_RIGHT[0] : WIDE_RIGHT[1]] == normalize(OLD_RIGHT)
    assert parent_rendered[WIDE_RAW_LEFT[0] : WIDE_RAW_LEFT[1]] == OLD_LEFT
    assert parent_rendered[WIDE_RAW_RIGHT[0] : WIDE_RAW_RIGHT[1]] == OLD_RIGHT
    left_emission = normalize(NEW_LEFT)
    right_emission = normalize(NEW_RIGHT)
    assert len(left_emission) == len(right_emission) == 52
    assert clause_boundaries(NEW_LEFT) == [13, 27, 52]
    assert clause_boundaries(NEW_RIGHT) == [13, 27, 39, 52]
    assert left_emission != right_emission[::-1]

    candidate_clauses = clause_parts(NEW_LEFT) + clause_parts(NEW_RIGHT)
    frames = [clause_frame(clause) for clause in candidate_clauses]
    frozen_frames = parent_frames(parent_rendered)
    global_frame_novelty = all(frame is not None and frame not in frozen_frames for frame in frames)
    connected_chain = ["Aram", "Liam", "Nadia", "Aram", "Ira", "Aidan", "Aram", "Nadia", "Aron"]
    neighboring_context = {
        "before_left": "Nora sees Aram.",
        "after_left": "Stressed was I, Aron.",
        "shared_boundary_entity": "Ira",
        "after_right_entity": "Aron",
    }
    bridge_gates = {
        "connected_entity_chain": True,
        "causal_temporal_anaphoric_link": True,
        "global_clause_novelty": all(clause not in clause_parts(parent_rendered) for clause in candidate_clauses),
        "global_frame_novelty": global_frame_novelty,
        "complete_english_clauses": all(clause_frame(clause) for clause in candidate_clauses),
        "catalogue_or_self_contained_unit": False,
        "neighboring_discourse_continuity": True,
    }
    assert bridge_gates["connected_entity_chain"]
    assert bridge_gates["causal_temporal_anaphoric_link"]
    assert bridge_gates["complete_english_clauses"]
    assert bridge_gates["catalogue_or_self_contained_unit"] is False
    assert bridge_gates["neighboring_discourse_continuity"]

    left_obligation = right_emission[::-1]
    right_obligation = left_emission[::-1]
    left_trace = compare_online(NEW_LEFT, left_obligation, "widened_left")
    right_trace = compare_online(NEW_RIGHT, right_obligation, "widened_right")
    assert not left_trace["exact"]
    assert not right_trace["exact"]
    assert left_trace["cursor"] == 0
    assert right_trace["cursor"] == 0

    attempt_rendered = (
        parent_rendered[: WIDE_RAW_LEFT[0]]
        + NEW_LEFT
        + " "
        + NEW_RIGHT
        + " "
        + parent_rendered[WIDE_RAW_RIGHT[1] :]
    )
    attempt_audit = audit(attempt_rendered)
    attempt_independent = independent_audit(attempt_rendered)
    assert attempt_independent["normalized_letters"] == 666
    assert not attempt_independent["two_pointer_exact"]

    row = {
        "id": "widened-bridge-no-closure-center-obstruction-666",
        "working_status": "widened_bridge_rejected_no_closure",
        "promotion_status": {
            "promoted": False,
            "status": "rejected_no_exact_closure",
            "reason": "The exact 10-letter center obstruction was persisted; one authored 52-letter connected bridge then contradicted at cursor 0 under reciprocal reversal.",
        },
        "rendered": parent_rendered,
        "audit": parent["audit"],
        "independent_audit": parent_audit,
        "parent_artifact": str(PARENT.relative_to(ROOT)),
        "parent_id": PARENT_ID,
        "parent_sha256": PARENT_SHA256,
        "growth_over_parent": 0,
        "center_obstruction": center_obstruction,
        "bridge_attempt": {
            "normalized_windows": {"left": list(WIDE_LEFT), "right": list(WIDE_RIGHT)},
            "raw_windows": {"left": list(WIDE_RAW_LEFT), "right": list(WIDE_RAW_RIGHT)},
            "old_left": OLD_LEFT,
            "old_right": OLD_RIGHT,
            "new_left": NEW_LEFT,
            "new_right": NEW_RIGHT,
            "clause_boundary_cursors": {"left": [13, 27, 52], "right": [13, 27, 39, 52]},
            "candidate_clauses": candidate_clauses,
            "connected_entity_chain": connected_chain,
            "neighboring_context": neighboring_context,
            "gates": bridge_gates,
            "left_emission": left_emission,
            "right_emission": right_emission,
            "left_obligation": left_obligation,
            "right_obligation": right_obligation,
            "left_trace": left_trace,
            "right_trace": right_trace,
            "attempt_rendered": attempt_rendered,
            "attempt_audit": attempt_audit,
            "attempt_independent_audit": attempt_independent,
            "final_residual": {"left": left_trace["residual"], "right": right_trace["residual"]},
            "exact_child_saved": False,
        },
        "provenance": "center 10-letter obstruction persisted before one widened authored four-event bridge; reciprocal closure contradicted immediately",
    }
    return {
        "experiment_id": "incumbent-666-center-obstruction-widened-bridge-20260922",
        "method": "persist center obstruction then one bounded connected four-event bridge",
        "active_frontier_parent": {"artifact": str(PARENT.relative_to(ROOT)), "id": PARENT_ID, "letters": 666, "sha256": PARENT_SHA256},
        "working_incumbent": {"artifact": "runs/incumbent-560-outer-causal-scene-20261002.json", "id": "outer-causal-scene-568-working-incumbent", "letters": 568, "sha256": "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"},
        "preserved_frontier": list(FRONTIER),
        "center_obstruction": center_obstruction,
        "rows": [row],
        "next_operator": "change to untouched actual seam normalized [197,281)/[385,469), raw [263,384)/[520,644); preserve 568 incumbent",
    }


def main() -> None:
    payload = build_payload()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["rows"][0]["bridge_attempt"]["left_trace"], sort_keys=True))
    print(payload["rows"][0]["promotion_status"])


if __name__ == "__main__":
    main()
