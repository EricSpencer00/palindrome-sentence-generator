#!/usr/bin/env python3
"""Reconstruct and reject a right-first outer-seam event proposal."""
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
OUT_PATH = ROOT / "runs" / "incumbent-568-outer-right-first-residual-20260923.json"
PARENT_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
PREFLIGHT_REVISION = "6d1fac0e"
LEFT_SPAN = (0, 48)
RIGHT_SPAN = (520, 568)
OLD_LEFT = "Leon won. Wolf spots Nora. Nadia stops, so Tara rewards Nadia"
OLD_RIGHT = "Aidan's drawer, Aratos, spots Aidan. Aron stops flow now, Noel"
NEW_LEFT = "A nut fell near Rhea while Otis opens the garden gate after dusk"
NEW_RIGHT = "Otis opens the garden gate while Rhea waits and finds fresh tuna"


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
                "first_mismatch": [left, tape[left], right, tape[right]],
                "comparisons": comparisons,
            }
        left += 1
        right -= 1
    return {"exact": True, "first_mismatch": None, "comparisons": comparisons}


def preflight(phrases: tuple[str, ...]) -> dict[str, Any]:
    command = ["git", "grep", "-n", "-F", "-i"]
    for phrase in phrases:
        command.extend(("-e", phrase))
    command.extend((PREFLIGHT_REVISION, "--", "runs", "experiments", "docs", "data"))
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if result.returncode not in (0, 1):
        raise RuntimeError(f"phrase-family preflight failed: {result.stderr}")
    hits = result.stdout.splitlines()
    return {
        "revision": PREFLIGHT_REVISION,
        "scope": "tracked runs/, experiments/, docs/, and data/",
        "phrases": list(phrases),
        "hits": hits,
        "status": "family_collision" if hits else "no_literal_collision",
    }


def build_payload() -> dict[str, Any]:
    source = json.loads(PARENT_PATH.read_text())
    parent_row = next(row for row in source["rows"] if row["working_status"] == "working_length_incumbent")
    parent = str(parent_row["rendered"])
    tape = normalize(parent)
    parent_hash = hashlib.sha256(tape.encode("ascii")).hexdigest()
    if len(tape) != 568 or parent_hash != PARENT_SHA256:
        raise AssertionError("pinned 568 parent changed")
    if tape[LEFT_SPAN[0] : LEFT_SPAN[1]] != normalize(OLD_LEFT):
        raise AssertionError("left source does not match its parent tape span")
    if tape[RIGHT_SPAN[0] : RIGHT_SPAN[1]] != normalize(OLD_RIGHT):
        raise AssertionError("right source does not match its parent tape span")
    if tape[LEFT_SPAN[0] : LEFT_SPAN[1]] != tape[RIGHT_SPAN[0] : RIGHT_SPAN[1]][::-1]:
        raise AssertionError("selected parent spans are not an exact reflected pair")

    left_start = parent.index(OLD_LEFT)
    left_end = left_start + len(OLD_LEFT)
    right_start = parent.index(OLD_RIGHT)
    right_end = right_start + len(OLD_RIGHT)
    rendered = parent[:left_start] + NEW_LEFT + parent[left_end:right_start] + NEW_RIGHT + parent[right_end:]
    child_tape = normalize(rendered)
    left_tape = normalize(NEW_LEFT)
    right_tape = normalize(NEW_RIGHT)
    required = right_tape[::-1]
    cursor = 0
    while cursor < min(len(left_tape), len(required)) and left_tape[cursor] == required[cursor]:
        cursor += 1
    scan = outside_in(child_tape)
    forward = hashlib.sha256(child_tape.encode("ascii")).hexdigest()
    reverse = hashlib.sha256(child_tape[::-1].encode("ascii")).hexdigest()
    project_exact = project_is_palindrome(rendered)
    phrase_audit = preflight((
        "A nut fell near Rhea while Otis opens the garden gate after dusk",
        "Otis opens the garden gate while Rhea waits and finds fresh tuna",
        "opens the garden gate",
        "Rhea waits and finds fresh tuna",
    ))

    return {
        "experiment_id": "incumbent-568-outer-right-first-residual-20260923",
        "working_status": "rejected_nonexact_repeated_event_phrase_collision",
        "parent": {
            "artifact": str(PARENT_PATH.relative_to(ROOT)),
            "letters": len(tape),
            "sha256_normalized": parent_hash,
        },
        "construction": {
            "operator": "right-first scene selection followed by left-side reverse-tape lexicalization",
            "proposer": "read-only Luna matrix-scene lane; independently reconstructed here",
            "normalized_parent_spans": [list(LEFT_SPAN), list(RIGHT_SPAN)],
            "raw_parent_spans": [[left_start, left_end], [right_start, right_end]],
            "replaced_left": OLD_LEFT,
            "replaced_right": OLD_RIGHT,
            "new_left": NEW_LEFT,
            "new_right": NEW_RIGHT,
        },
        "candidate": {
            "rendered": rendered,
            "letters": len(child_tape),
            "growth_over_parent": len(child_tape) - len(tape),
            "normalized_sha256": forward,
            "audit": {
                "independent_outside_in_exact": scan["exact"],
                "first_mismatch": scan["first_mismatch"],
                "project_validator_exact": project_exact,
                "sha256_forward": forward,
                "sha256_reverse": reverse,
                "hashes_equal": forward == reverse,
            },
            "local_equation": {
                "left_tape": left_tape,
                "right_tape": right_tape,
                "required_reverse_right_tape": required,
                "left_letters": len(left_tape),
                "right_letters": len(right_tape),
                "matched_prefix_letters": cursor,
                "first_mismatch": {
                    "cursor": cursor,
                    "left": left_tape[cursor],
                    "required": required[cursor],
                },
            },
            "same_turn_repeated_event": {
                "surface": "Otis opens the garden gate",
                "occurrences_in_insertions": 2,
                "status": "rejected_repetition_shortcut",
            },
        },
        "novelty_preflight": phrase_audit,
        "admission": {
            "admitted": False,
            "reason": "The live equation fails after `a nut`; insertion lengths differ by one letter, the same Otis/garden-gate event is repeated on both sides, and the phrase family collides with prior scene runs.",
            "readability_certified": False,
            "reader_evidence": False,
        },
        "next_action": "Retire this outer seam and garden-gate event frame. Preflight new participant and predicate families before drafting; require equal insertion lengths and live residual closure before semantic review.",
    }


if __name__ == "__main__":
    payload = build_payload()
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "status": payload["working_status"],
        "letters": payload["candidate"]["letters"],
        "sha256": payload["candidate"]["normalized_sha256"],
        "left_right_insertion_lengths": [
            payload["candidate"]["local_equation"]["left_letters"],
            payload["candidate"]["local_equation"]["right_letters"],
        ],
        "cursor": payload["candidate"]["local_equation"]["matched_prefix_letters"],
        "novelty_status": payload["novelty_preflight"]["status"],
    }, indent=2))
