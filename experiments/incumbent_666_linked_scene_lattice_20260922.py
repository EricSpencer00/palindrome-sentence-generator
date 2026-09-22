"""Bounded seam-local typed production on the promoted 666 parent.

This experiment adds exactly one transitive family from the existing lexical
inventory (``writes``). It emits a few authored typed event graphs, carrying
the reverse residual, active entity, global novelty sets, and raw shell spaces
at every paired cursor. It is not a vocabulary or whole-sentence sweep.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.incumbent_666_boundary_discourse_linker_20260922 import (
    FRONTIER,
    independent_audit,
    normalize,
    validate_frontier_entry,
)
from experiments.incumbent_666_reciprocal_paired_production_20260922 import extract_frames


PARENT = ROOT / "runs" / "incumbent-666-context-aware-reciprocal-pair-20260922.json"
OUT = ROOT / "runs" / "incumbent-666-linked-scene-lattice-20260922.json"
PARENT_ID = "context-aware-reciprocal-pair-noel-sara-666"
PARENT_SHA256 = "a1b4ebaaba06893fdfa2a495676b887355e361266b2960a61818981e59e0da37"
NORMALIZED_LEFT = (140, 212)
NORMALIZED_RIGHT = (454, 526)
RAW_LEFT = (187, 285)
RAW_RIGHT = (621, 718)
ALTERNATE_NORMALIZED_LEFT = (127, 140)
ALTERNATE_NORMALIZED_RIGHT = (526, 539)
ALTERNATE_RAW_LEFT = (171, 188)
ALTERNATE_RAW_RIGHT = (719, 736)
MAX_PAIRED_SCENE_PRODUCTIONS = 8

# Exactly one newly supported family, sourced from the existing BUNDLES/
# lexical inventory in two_clause_joint_seam_bundle_search_20260917.py.
TRANSITIVE_FAMILY = {
    "predicate": "writes",
    "source": "experiments/two_clause_joint_seam_bundle_search_20260917.py:BUNDLES",
    "valency": "subject-writes-object",
}

# Four authored, finite event graphs are the complete bounded lattice. Every
# side is 72 letters, connected through the active entity, and uses only the
# new family so global frame novelty can be checked independently.
SCENE_PRODUCTIONS = (
    {
        "id": "writes-chain-nora-mara",
        "left": "Nora writes Ari. Ari writes Aram. Aram writes Nadia. Nadia writes Aidan. Aidan writes Mara.",
        "right": "Mara writes Aidan. Aidan writes Nadia. Nadia writes Aram. Aram writes Ari. Ari writes Nora.",
    },
    {
        "id": "writes-chain-sara-mara",
        "left": "Sara writes Ari. Ari writes Aram. Aram writes Nadia. Nadia writes Aidan. Aidan writes Mara.",
        "right": "Mara writes Aidan. Aidan writes Nadia. Nadia writes Aram. Aram writes Ari. Ari writes Sara.",
    },
    {
        "id": "writes-chain-nora-aras",
        "left": "Nora writes Ari. Ari writes Aras. Aras writes Nadia. Nadia writes Aidan. Aidan writes Mara.",
        "right": "Mara writes Aidan. Aidan writes Nadia. Nadia writes Aras. Aras writes Ari. Ari writes Nora.",
    },
    {
        "id": "writes-chain-nora-sara",
        "left": "Nora writes Aram. Aram writes Nadia. Nadia writes Aidan. Aidan writes Ari. Ari writes Mara.",
        "right": "Mara writes Ari. Ari writes Aidan. Aidan writes Nadia. Nadia writes Aram. Aram writes Nora.",
    },
)


def clauses(text: str) -> list[str]:
    return [part.strip() + "." for part in re.split(r"[.!?;]+", text) if part.strip()]


def typed_events(text: str) -> list[dict[str, str]]:
    pattern = re.compile(r"\b([A-Z][a-z]+) writes ([A-Z][a-z]+)\.")
    return [{"subject": s, "predicate": "writes", "object": o} for s, o in pattern.findall(text)]


def typed_frames(text: str) -> set[str]:
    return {f"{event['subject'].lower()}|{event['predicate']}" for event in typed_events(text)}


def online_pair(production: dict[str, str], parent_frames: set[str], parent_clauses: set[str], shell: dict[str, str]) -> dict[str, object]:
    left = production["left"]
    right = production["right"]
    left_tape = normalize(left)
    right_tape = normalize(right)
    expected_right = left_tape[::-1]
    left_events = typed_events(left)
    candidate_frames = typed_frames(left) | typed_frames(right)
    candidate_clauses = {clause.lower() for clause in clauses(left + " " + right)}
    reused_frames = sorted(candidate_frames & parent_frames)
    reused_clauses = sorted(candidate_clauses & parent_clauses)
    trace: list[dict[str, object]] = []
    active_entity = shell["left_before"]
    used_frames: set[str] = set()
    used_clauses: set[str] = set()
    clause_ranges: list[tuple[int, int, str, str]] = []
    clause_cursor = 0
    for event in left_events:
        clause = f"{event['subject']} writes {event['object']}."
        clause_tape = normalize(clause)
        clause_ranges.append((clause_cursor, clause_cursor + len(clause_tape), clause.lower(), f"{event['subject'].lower()}|writes"))
        clause_cursor += len(clause_tape)
    for cursor, emitted_left in enumerate(left_tape):
        emitted_right = right_tape[cursor] if cursor < len(right_tape) else None
        owner_start, owner_end, clause_owner, frame_owner = next(item for item in clause_ranges if item[0] <= cursor < item[1])
        active_entity_before = active_entity
        used_clauses.add(clause_owner)
        used_frames.add(frame_owner)
        active_entity = left_events[len(used_clauses) - 1]["object"]
        state = {
            "cursor": cursor,
            "reverse_residual": expected_right[cursor:],
            "emitted_left": emitted_left,
            "emitted_right": emitted_right,
            "expected_right": expected_right[cursor],
            "active_entity": active_entity,
            "active_entity_before": active_entity_before,
            "owner": "typed-writes-production",
            "clause_owner": clause_owner,
            "clause_cursor": [owner_start, owner_end],
            "leading_shell_space": shell["left_leading"],
            "trailing_shell_space": shell["left_trailing"],
            "used_frames": sorted(used_frames),
            "used_clauses": sorted(used_clauses),
        }
        trace.append(state)
        if emitted_right != expected_right[cursor]:
            return {
                "exact": False,
                "cursor": cursor,
                "expected": expected_right[cursor],
                "emitted": emitted_right,
                "residual": expected_right[cursor:],
                "owner": "reverse-residual",
                "grammar_state": {
                    "active_entity": active_entity,
                    "candidate_frames": sorted(candidate_frames),
                    "reused_frames": reused_frames,
                    "reused_clauses": reused_clauses,
                    "used_frames": sorted(used_frames),
                    "used_clauses": sorted(used_clauses),
                },
                "trace": trace,
            }
    return {
        "exact": left_tape == right_tape[::-1],
        "cursor": len(left_tape),
        "expected": None,
        "emitted": None,
        "residual": "",
        "owner": "reverse-residual",
        "grammar_state": {"active_entity": active_entity, "candidate_frames": sorted(candidate_frames), "reused_frames": reused_frames, "reused_clauses": reused_clauses},
        "trace": trace,
    }


def assemble(parent: str, left: str, right: str, raw_left: tuple[int, int], raw_right: tuple[int, int]) -> tuple[str, dict[str, object]]:
    left_leading = parent[raw_left[0]]
    right_leading = parent[raw_right[0]]
    left_trailing = parent[raw_left[1]]
    right_trailing = parent[raw_right[1]]
    assert left_leading == right_leading == left_trailing == right_trailing == " "
    rendered = parent[: raw_left[0]] + left_leading + left + parent[raw_left[1] : raw_right[0]] + right_leading + right + parent[raw_right[1] :]
    left_start = raw_left[0]
    left_end = left_start + len(left_leading) + len(left)
    left_delta = len(left_leading) + len(left) - (raw_left[1] - raw_left[0])
    right_start = raw_right[0] + left_delta
    right_end = right_start + len(right_leading) + len(right)
    spacing = {
        "left_leading": left_leading,
        "right_leading": right_leading,
        "left_trailing": left_trailing,
        "right_trailing": right_trailing,
        "left_prefix_excerpt": rendered[left_start - 12 : left_start + 18],
        "right_prefix_excerpt": rendered[right_start - 12 : right_start + 18],
        "leading_shell_spaces_preserved": rendered[left_start - 1 : left_start + 2] == ". N" and rendered[right_start - 1 : right_start + 2] == ". M",
        "trailing_shell_spaces_preserved": rendered[left_end - 1 : left_end + 2].startswith(". ") and rendered[right_end - 1 : right_end + 2].startswith(". "),
    }
    spacing["spacing_shell_preserved"] = bool(spacing["leading_shell_spaces_preserved"] and spacing["trailing_shell_spaces_preserved"])
    return rendered, spacing


def build_payload() -> dict[str, object]:
    payload = json.loads(PARENT.read_text())
    parent_row = next(row for row in payload["rows"] if row["id"] == PARENT_ID)
    parent_rendered = str(parent_row["rendered"])
    parent_audit = independent_audit(parent_rendered)
    assert parent_audit["normalized_letters"] == 666 and parent_audit["two_pointer_exact"]
    assert parent_audit["sha256_forward"] == PARENT_SHA256
    assert parent_row["promotion_status"]["promoted"] is True
    for entry in FRONTIER:
        validate_frontier_entry(entry)
    parent_frames = extract_frames(parent_rendered) | typed_frames(parent_rendered)
    parent_clauses = {clause.lower() for clause in clauses(parent_rendered)}
    target = normalize(parent_rendered)
    assert target[NORMALIZED_LEFT[0] : NORMALIZED_LEFT[1]] == target[NORMALIZED_RIGHT[0] : NORMALIZED_RIGHT[1]][::-1]
    assert len(target[NORMALIZED_LEFT[0] : NORMALIZED_LEFT[1]]) == 72

    shell = {"left_before": "Noel", "right_before": "Leon", "left_leading": " ", "left_trailing": " ", "right_leading": " ", "right_trailing": " "}
    attempts: list[dict[str, object]] = []
    for production in SCENE_PRODUCTIONS:
        left_tape = normalize(production["left"])
        right_tape = normalize(production["right"])
        assert len(left_tape) == len(right_tape) == 72
        attempt = online_pair(production, parent_frames, parent_clauses, shell)
        attempt.update({"id": production["id"], "letters_per_side": 72, "complete_scene": len(typed_events(production["left"])) == 5 and len(typed_events(production["right"])) == 5, "candidate_clause_count": 10})
        attempts.append(attempt)
    primary = attempts[0]
    candidate_rendered, spacing = assemble(parent_rendered, SCENE_PRODUCTIONS[0]["left"], SCENE_PRODUCTIONS[0]["right"], RAW_LEFT, RAW_RIGHT)
    candidate_audit = independent_audit(candidate_rendered)
    assert candidate_audit["normalized_letters"] == 666
    assert candidate_audit["two_pointer_exact"] is False
    primary["spacing"] = spacing
    primary["exact_candidate_audit"] = candidate_audit
    primary["admission"] = {"accepted": False, "exact_character_closure": False, "exact_child_saved": False, "reason": "writes residual contradicts the natural typed reverse stream"}

    alternate_production = {"left": "Nora writes Ari.", "right": "Ira writes Aron."}
    alternate_attempt = online_pair(alternate_production, parent_frames, parent_clauses, shell)
    alternate_attempt.update({"id": "alternate-13-letter-writes-seam", "normalized_windows": {"left": list(ALTERNATE_NORMALIZED_LEFT), "right": list(ALTERNATE_NORMALIZED_RIGHT)}, "raw_windows": {"left": list(ALTERNATE_RAW_LEFT), "right": list(ALTERNATE_RAW_RIGHT)}, "letters_per_side": 13})
    row = {
        "id": "typed-writes-lattice-no-admission-666",
        "working_status": "typed_writes_rejected_residual_and_spacing_gate",
        "promotion_status": {"promoted": False, "status": "rejected_typed_reverse_residual", "reason": "no natural writes production crossed the reverse residual; no exact child admitted"},
        "rendered": parent_rendered,
        "independent_audit": parent_audit,
        "parent_artifact": str(PARENT.relative_to(ROOT)),
        "parent_id": PARENT_ID,
        "parent_sha256": PARENT_SHA256,
        "growth_over_parent": 0,
        "typed_production_attempt": {
            "normalized_windows": {"left": list(NORMALIZED_LEFT), "right": list(NORMALIZED_RIGHT)},
            "raw_windows": {"left": list(RAW_LEFT), "right": list(RAW_RIGHT)},
            "predicate_family": TRANSITIVE_FAMILY,
            "production_count": len(SCENE_PRODUCTIONS),
            "max_productions": MAX_PAIRED_SCENE_PRODUCTIONS,
            "attempts": attempts,
            "primary_candidate_rendered": candidate_rendered,
            "primary_candidate_sha256": candidate_audit["sha256_forward"],
            "primary_spacing": spacing,
            "alternate_seam_attempt": alternate_attempt,
            "shell_ownership": shell,
            "admission": primary["admission"],
        },
        "next_operator": "switch to actual alternate seam [127,140]↔[526,539] raw [171,188]↔[719,736] after typed writes residual obstruction",
        "provenance": "one seam-local typed writes family with online reverse residual, entity, novelty, and shell-space tracking",
    }
    return {
        "experiment_id": "incumbent-666-linked-scene-lattice-20260922",
        "method": "bounded typed finite-event writes production on promoted 72-letter seam, then one actual 13-letter alternate seam obstruction",
        "active_frontier_parent": {"artifact": str(PARENT.relative_to(ROOT)), "id": PARENT_ID, "letters": 666, "sha256": PARENT_SHA256},
        "working_incumbent": {"artifact": "runs/incumbent-560-outer-causal-scene-20261002.json", "id": "outer-causal-scene-568-working-incumbent", "letters": 568, "sha256": "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"},
        "preserved_frontier": list(FRONTIER),
        "rows": [row],
        "next_operator": row["next_operator"],
    }


def main() -> None:
    result = build_payload()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    attempt = result["rows"][0]["typed_production_attempt"]
    print({"id": result["rows"][0]["id"], "productions": attempt["production_count"], "cursor": attempt["attempts"][0]["cursor"], "alternate_cursor": attempt["alternate_seam_attempt"]["cursor"]})


if __name__ == "__main__":
    main()
