#!/usr/bin/env python3
"""Reconstruct and reject a 568-parent complete-boundary scene splice.

This is an audit artifact: the candidate is kept visible, but is not admitted
because its live character residual fails and an authored event phrase
collides with an older experiment.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.validator import is_palindrome as project_is_palindrome

PARENT_PATH = ROOT / "runs" / "incumbent-560-outer-causal-scene-20261002.json"
OUT_PATH = ROOT / "runs" / "incumbent-568-boundary-event-residual-20260923.json"
PARENT_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
PREFLIGHT_REVISION = "fffb2af8178c9541c521e7738fa1b2d1b96f3f34"
LEFT_SPAN = (186, 194)
RIGHT_SPAN = (374, 382)
OLD_LEFT = "Pat notes"
OLD_RIGHT = "Seton, tap"
NEW_LEFT = "Aron carried a lantern; Nora waited"
NEW_RIGHT = "Ari heard a bell near Nora"


def normalize(text: str) -> str:
    return "".join(char.lower() for char in text if char.isascii() and char.isalpha())


def outside_in(tape: str) -> dict[str, Any]:
    left, right = 0, len(tape) - 1
    comparisons = 0
    while left < right:
        comparisons += 1
        if tape[left] != tape[right]:
            return {
                "exact": False,
                "first_mismatch": {
                    "left_cursor": left,
                    "right_cursor": right,
                    "left_char": tape[left],
                    "right_char": tape[right],
                },
                "comparisons": comparisons,
            }
        left += 1
        right -= 1
    return {"exact": True, "first_mismatch": None, "comparisons": comparisons}


def literal_preflight(phrases: tuple[str, ...]) -> dict[str, Any]:
    command = ["git", "grep", "-n", "-F", "-i"]
    for phrase in phrases:
        command.extend(("-e", phrase))
    command.extend((PREFLIGHT_REVISION, "--", "runs", "experiments", "docs", "data"))
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if result.returncode not in (0, 1):
        raise RuntimeError(f"novelty preflight failed: {result.stderr}")
    hits = [line for line in result.stdout.splitlines()]
    return {
        "revision": PREFLIGHT_REVISION,
        "scope": "tracked runs/, experiments/, docs/, and data/",
        "phrases": list(phrases),
        "hits": hits,
        "status": "collision" if hits else "no_literal_collision",
    }


def build_payload() -> dict[str, Any]:
    parent_payload = json.loads(PARENT_PATH.read_text())
    parent_row = next(
        row for row in parent_payload["rows"]
        if row["working_status"] == "working_length_incumbent"
    )
    parent = str(parent_row["rendered"])
    parent_tape = normalize(parent)
    parent_hash = hashlib.sha256(parent_tape.encode("ascii")).hexdigest()
    if len(parent_tape) != 568 or parent_hash != PARENT_SHA256:
        raise AssertionError("pinned 568 parent changed")

    if parent_tape[LEFT_SPAN[0] : LEFT_SPAN[1]] != normalize(OLD_LEFT):
        raise AssertionError("left source does not match its normalized span")
    if parent_tape[RIGHT_SPAN[0] : RIGHT_SPAN[1]] != normalize(OLD_RIGHT):
        raise AssertionError("right source does not match its normalized span")
    if parent_tape[LEFT_SPAN[0] : LEFT_SPAN[1]] != parent_tape[RIGHT_SPAN[0] : RIGHT_SPAN[1]][::-1]:
        raise AssertionError("selected source spans are not the parent's exact reflected pair")

    left_start = parent.index(OLD_LEFT)
    right_start = parent.index(OLD_RIGHT)
    left_end = left_start + len(OLD_LEFT)
    right_end = right_start + len(OLD_RIGHT)
    if parent[left_end] != "." or parent[right_end] != ".":
        raise AssertionError("sentence terminators expected to remain outside the word spans")
    rendered = (
        parent[:left_start]
        + NEW_LEFT
        + parent[left_end:right_start]
        + NEW_RIGHT
        + parent[right_end:]
    )
    tape = normalize(rendered)
    left_tape = normalize(NEW_LEFT)
    right_tape = normalize(NEW_RIGHT)
    required_right_tape = left_tape[::-1]
    local_cursor = 0
    while (
        local_cursor < min(len(left_tape), len(right_tape))
        and left_tape[local_cursor] == right_tape[::-1][local_cursor]
    ):
        local_cursor += 1
    pointer = outside_in(tape)
    forward_hash = hashlib.sha256(tape.encode("ascii")).hexdigest()
    reverse_hash = hashlib.sha256(tape[::-1].encode("ascii")).hexdigest()
    project_exact = project_is_palindrome(rendered)
    novelty = literal_preflight(("Aron carried a lantern", "Nora waited", "Ari heard a bell near Nora"))

    return {
        "experiment_id": "incumbent-568-boundary-event-residual-20260923",
        "working_status": "rejected_nonexact_and_phrase_collision",
        "parent": {
            "artifact": str(PARENT_PATH.relative_to(ROOT)),
            "letters": len(parent_tape),
            "sha256_normalized": parent_hash,
        },
        "method": {
            "name": "complete-sentence-boundary event insertion with live reversed-tape obligation",
            "lane": "read-only Luna matrix-scene proposal, independently reconstructed here",
            "geometry": "new event on the left; distinct auditory-near-location event on the right",
        },
        "edit": {
            "normalized_parent_spans": [list(LEFT_SPAN), list(RIGHT_SPAN)],
            "raw_parent_spans": [[left_start, left_end], [right_start, right_end]],
            "replaced_left": OLD_LEFT,
            "replaced_right": OLD_RIGHT,
            "new_left": NEW_LEFT,
            "new_right": NEW_RIGHT,
            "left_letters": len(left_tape),
            "right_letters": len(right_tape),
            "retained_middle_unchanged": True,
        },
        "candidate": {
            "rendered": rendered,
            "letters": len(tape),
            "growth_over_parent": len(tape) - len(parent_tape),
            "normalized_sha256": forward_hash,
            "exact_audit": {
                "independent_outside_in_exact": pointer["exact"],
                "outside_in_first_mismatch": pointer["first_mismatch"],
                "project_validator_exact": project_exact,
                "sha256_forward": forward_hash,
                "sha256_reverse": reverse_hash,
                "hashes_equal": forward_hash == reverse_hash,
            },
            "local_residual": {
                "left_tape": left_tape,
                "actual_reverse_right_tape": right_tape[::-1],
                "required_reverse_right_tape": required_right_tape,
                "matched_prefix_letters": local_cursor,
                "first_mismatch": {
                    "cursor": local_cursor,
                    "left": left_tape[local_cursor],
                    "required": right_tape[::-1][local_cursor],
                },
            },
        },
        "novelty_preflight": novelty,
        "repair_attempt": {
            "operator": "freeze one event tape and word-lattice segment its exact character reverse into a distinct event frame",
            "frozen_left_tape": left_tape,
            "required_reverse_tape": required_right_tape,
            "lane_report": {
                "viable_prefixes": ["de", "det"],
                "dead_residual_after_det": "iawaronnretnaladeirracnora",
                "complete_grammatical_parses": 0,
                "independently_reproduced": False,
            },
        },
        "parallel_lane_reports": [
            {
                "lane": "typed-auxiliary-relative-attachment-chart",
                "normalized_spans": [[112, 148], [420, 456]],
                "fresh_units": 200,
                "chart_states": 15236,
                "long_path_attempts": 15036,
                "exact_closures": 0,
                "first_character_compatible_opposing_state": False,
                "source": "read-only Luna lane report; not independently rerun in this artifact",
            },
            {
                "lane": "perception-motion-possession-event-chart",
                "normalized_spans": [[20, 48], [520, 548]],
                "fresh_event_units": 384,
                "chart_states": 12640,
                "long_path_attempts": 4066,
                "exact_closures": 0,
                "furthest_live_prefix_letters": 5,
                "reported_attempt": "Nadia hears Nora. Nora carries Nora.",
                "source": "read-only Luna lane report; not independently rerun in this artifact",
            },
        ],
        "same_turn_duplicate_attempts": [
            {
                "lane": "auxiliary-participle-valency-chart",
                "normalized_spans": [[186, 194], [374, 382]],
                "fresh_units_reported": 2304,
                "exact_closures_reported": 0,
                "attempt_reported": "Mara has carried Mara.",
                "local_residual_reported": "ahascarriedmara",
                "preflight_decision": "rejected_as_duplicate_geometry",
                "duplicate_of": "this experiment's [186,194)/[374,382) carried-object residual probe",
                "source": "late same-turn Luna lane report; geometry predates this run and is not admitted as a fresh experiment",
            }
        ],
        "admission": {
            "admitted": False,
            "reason": "the local character equation fails after four characters; independent novelty preflight also finds `Nora waited` in endpoint_seeded_scene_inward_20260920.py",
            "readability_certified": False,
            "reader_evidence": False,
        },
        "next_action": "Retire this seam. Continue at a different complete-sentence-boundary pair with fresh participant/event roles, and run phrase-family novelty before composing either side.",
    }


if __name__ == "__main__":
    payload = build_payload()
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "status": payload["working_status"],
        "letters": payload["candidate"]["letters"],
        "sha256": payload["candidate"]["normalized_sha256"],
        "local_matched_prefix": payload["candidate"]["local_residual"]["matched_prefix_letters"],
        "novelty_status": payload["novelty_preflight"]["status"],
    }, indent=2))
