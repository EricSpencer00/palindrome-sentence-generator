#!/usr/bin/env python3
"""Reconstruct and reject a 24-letter attachment pair on the 568 tape."""
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
OUT_PATH = ROOT / "runs" / "audit-incumbent-568-attachment-pair-residual-20260923.json"
PARENT_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
PREFLIGHT_REVISION = "be9191c4"
LEFT_SPAN = (204, 220)
RIGHT_SPAN = (348, 364)
NEW_LEFT = "Milo marks a quiet map at dawn"
NEW_RIGHT = "Nora reads a calm path at dusk"


def normalize(text: str) -> str:
    return "".join(char.lower() for char in text if char.isascii() and char.isalpha())


def raw_boundary_for_letter_offset(text: str, letter_offset: int) -> int:
    count = 0
    if letter_offset == 0:
        return 0
    for raw_offset, char in enumerate(text):
        if char.isascii() and char.isalpha():
            count += 1
            if count == letter_offset:
                return raw_offset + 1
    raise ValueError(f"letter offset {letter_offset} is outside this tape")


def raw_start_for_letter_offset(text: str, letter_offset: int) -> int:
    raw_offset = raw_boundary_for_letter_offset(text, letter_offset)
    while raw_offset < len(text) and not (text[raw_offset].isascii() and text[raw_offset].isalpha()):
        raw_offset += 1
    return raw_offset


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


def phrase_preflight(phrases: tuple[str, ...]) -> dict[str, Any]:
    command = ["git", "grep", "-n", "-F", "-i"]
    for phrase in phrases:
        command.extend(("-e", phrase))
    command.extend((PREFLIGHT_REVISION, "--", "runs", "experiments", "docs", "data"))
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if result.returncode not in (0, 1):
        raise RuntimeError(f"phrase preflight failed: {result.stderr}")
    hits = result.stdout.splitlines()
    return {
        "revision": PREFLIGHT_REVISION,
        "phrases": list(phrases),
        "scope": "tracked runs/, experiments/, docs/, and data/",
        "hits": hits,
        "status": "no_literal_hits" if not hits else "phrase_family_collision",
    }


def build_payload() -> dict[str, Any]:
    source = json.loads(PARENT_PATH.read_text())
    row = next(item for item in source["rows"] if item["working_status"] == "working_length_incumbent")
    parent = str(row["rendered"])
    parent_tape = normalize(parent)
    parent_sha = hashlib.sha256(parent_tape.encode("ascii")).hexdigest()
    if len(parent_tape) != 568 or parent_sha != PARENT_SHA256:
        raise AssertionError("pinned 568 parent changed")

    left_raw = (
        raw_start_for_letter_offset(parent, LEFT_SPAN[0]),
        raw_boundary_for_letter_offset(parent, LEFT_SPAN[1]),
    )
    right_raw = (
        raw_start_for_letter_offset(parent, RIGHT_SPAN[0]),
        raw_boundary_for_letter_offset(parent, RIGHT_SPAN[1]),
    )
    left_source = parent_tape[LEFT_SPAN[0] : LEFT_SPAN[1]]
    right_source = parent_tape[RIGHT_SPAN[0] : RIGHT_SPAN[1]]
    if len(left_source) != 16 or left_source != right_source[::-1]:
        raise AssertionError("the selected 16-letter parent seam is not exactly reflected")
    if not left_raw[1] <= right_raw[0]:
        raise AssertionError("replacement windows overlap in the rendered parent")

    left_tape, right_tape = normalize(NEW_LEFT), normalize(NEW_RIGHT)
    if len(left_tape) != 24 or len(right_tape) != 24:
        raise AssertionError("the bounded attempt requires exactly 24 letters on each side")
    required_left = right_tape[::-1]
    cursor = 0
    while cursor < len(left_tape) and left_tape[cursor] == required_left[cursor]:
        cursor += 1

    rendered = (
        parent[: left_raw[0]]
        + NEW_LEFT
        + parent[left_raw[1] : right_raw[0]]
        + NEW_RIGHT
        + parent[right_raw[1] :]
    )
    tape = normalize(rendered)
    independent_scan = outside_in(tape)
    project_exact = project_is_palindrome(rendered)
    forward = hashlib.sha256(tape.encode("ascii")).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode("ascii")).hexdigest()
    if independent_scan["exact"] or project_exact or forward == reverse:
        raise AssertionError("the reported zero-prefix obstruction unexpectedly closed")
    if cursor != 0:
        raise AssertionError("reported first-character obstruction changed")

    return {
        "experiment_id": "audit-incumbent-568-attachment-pair-residual-20260923",
        "working_status": "rejected_nonexact_zero_prefix_context_invalid",
        "parent": {
            "artifact": str(PARENT_PATH.relative_to(ROOT)),
            "letters": len(parent_tape),
            "sha256_normalized": parent_sha,
        },
        "operator": {
            "name": "equal-length independent attachment-pair probe with live reverse residual",
            "provenance": "bounded Luna matrix-scene lane; parent, raw offsets, child, and exactness independently reconstructed here",
            "preflight_revision": PREFLIGHT_REVISION,
            "normalized_spans": [list(LEFT_SPAN), list(RIGHT_SPAN)],
            "raw_spans": [list(left_raw), list(right_raw)],
            "source_blocks": [parent[left_raw[0] : left_raw[1]], parent[right_raw[0] : right_raw[1]]],
            "insertions": [NEW_LEFT, NEW_RIGHT],
        },
        "candidate": {
            "rendered": rendered,
            "letters": len(tape),
            "growth_over_parent": len(tape) - len(parent_tape),
            "normalized_sha256": forward,
            "audit": {
                "independent_outside_in_exact": independent_scan["exact"],
                "first_mismatch": independent_scan["first_mismatch"],
                "project_validator_exact": project_exact,
                "sha256_forward": forward,
                "sha256_reverse": reverse,
                "hashes_equal": forward == reverse,
            },
            "local_equation": {
                "left_tape": left_tape,
                "right_tape": right_tape,
                "required_reverse_right_tape": required_left,
                "left_letters": len(left_tape),
                "right_letters": len(right_tape),
                "matched_prefix_letters": cursor,
                "first_mismatch": {"cursor": 0, "left": left_tape[0], "required": required_left[0]},
                "left_residual": left_tape,
                "required_right_residual": required_left,
            },
            "context_check": {
                "left_after_splice": "Mara saw God. Milo marks a quiet map at dawn, I saw desserts.",
                "right_after_splice": "Stressed was I, Nora reads a calm path at dusk. Dog was Aram.",
                "status": "both inserted sentences are locally readable, but neither attaches grammatically to its retained flank",
            },
        },
        "novelty_preflight": phrase_preflight((NEW_LEFT, NEW_RIGHT)),
        "admission": {
            "admitted": False,
            "reason": "Although the 24/24 insertions would add 16 letters, their first characters conflict (`m/k`), so the live equation has zero compatible characters. The rendered diagnostic also creates a left comma splice and an incompatible right appositive/clause after the retained flanks.",
            "human_readability_certified": False,
            "reader_evidence": False,
        },
        "next_construction": "Change the actual seam and pair a boundary-compatible opening/ending inside a connected two-clause discourse skeleton; enforce both flank parses, equal insertion lengths, and the first residual character before drafting full phrases.",
    }


if __name__ == "__main__":
    payload = build_payload()
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    candidate = payload["candidate"]
    print(json.dumps({
        "status": payload["working_status"],
        "letters": candidate["letters"],
        "growth": candidate["growth_over_parent"],
        "sha256": candidate["normalized_sha256"],
        "insert_lengths": [candidate["local_equation"]["left_letters"], candidate["local_equation"]["right_letters"]],
        "matched_prefix": candidate["local_equation"]["matched_prefix_letters"],
        "first_mismatch": candidate["audit"]["first_mismatch"],
        "phrase_preflight": payload["novelty_preflight"]["status"],
    }, indent=2))
