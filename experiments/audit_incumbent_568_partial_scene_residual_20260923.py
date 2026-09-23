#!/usr/bin/env python3
"""Reconstruct a partial-word scene proposal and retain its live obstruction."""
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
OUT_PATH = ROOT / "runs" / "audit-incumbent-568-partial-scene-residual-20260923.json"
PARENT_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
PREFLIGHT_REVISION = "f85ef046"
LEFT_SPAN = (222, 248)
RIGHT_SPAN = (320, 346)
OLD_LEFT = "aw desserts. Noel, was I stressed"
OLD_RIGHT = "Desserts I saw, Leon. Stressed wa"
NEW_LEFT = "aw Mina carry copper cask for Niko"
NEW_RIGHT = "Wolf weighs this amber vial; witness wa"
EXPECTED_SHA256 = "e83ac3f3d51d6624f2547bafffe6b76ef07f4e8313cfae05e983786ccdc50697"


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


def phrase_preflight(phrases: tuple[str, ...]) -> dict[str, Any]:
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
    parent_hash = hashlib.sha256(parent_tape.encode("ascii")).hexdigest()
    if len(parent_tape) != 568 or parent_hash != PARENT_SHA256:
        raise AssertionError("pinned 568 parent changed")
    if parent_tape[LEFT_SPAN[0] : LEFT_SPAN[1]] != normalize(OLD_LEFT):
        raise AssertionError("left source fragment does not match parent geometry")
    if parent_tape[RIGHT_SPAN[0] : RIGHT_SPAN[1]] != normalize(OLD_RIGHT):
        raise AssertionError("right source fragment does not match parent geometry")
    if parent_tape[LEFT_SPAN[0] : LEFT_SPAN[1]] != parent_tape[RIGHT_SPAN[0] : RIGHT_SPAN[1]][::-1]:
        raise AssertionError("parent fragments are not a reflected pair")
    typed_left_span = (73, 100)
    typed_right_span = (468, 495)
    if parent_tape[typed_left_span[0] : typed_left_span[1]] != parent_tape[typed_right_span[0] : typed_right_span[1]][::-1]:
        raise AssertionError("typed-chart report geometry is not an exact reflected pair")
    morphology_left_span = (250, 259)
    morphology_right_span = (309, 318)
    if parent_tape[morphology_left_span[0] : morphology_left_span[1]] != parent_tape[morphology_right_span[0] : morphology_right_span[1]][::-1]:
        raise AssertionError("morphology-chart report geometry is not an exact reflected pair")

    left_start = parent.index(OLD_LEFT)
    left_end = left_start + len(OLD_LEFT)
    right_start = parent.index(OLD_RIGHT)
    right_end = right_start + len(OLD_RIGHT)
    rendered = parent[:left_start] + NEW_LEFT + parent[left_end:right_start] + NEW_RIGHT + parent[right_end:]
    tape = normalize(rendered)
    left_tape = normalize(NEW_LEFT)
    right_tape = normalize(NEW_RIGHT)
    required = right_tape[::-1]
    cursor = 0
    while cursor < min(len(left_tape), len(required)) and left_tape[cursor] == required[cursor]:
        cursor += 1
    scan = outside_in(tape)
    forward = hashlib.sha256(tape.encode("ascii")).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode("ascii")).hexdigest()
    project_exact = project_is_palindrome(rendered)
    if forward != EXPECTED_SHA256:
        raise AssertionError("reconstructed child differs from the lane's normalized hash")
    if scan["exact"] or project_exact or forward == reverse:
        raise AssertionError("the reported nonexact child unexpectedly closes")
    preflight = phrase_preflight((
        "Mina carry copper cask for Niko",
        "Wolf weighs this amber vial",
        "copper cask",
        "amber vial",
        "witness was I, Aron",
    ))

    return {
        "experiment_id": "audit-incumbent-568-partial-scene-residual-20260923",
        "working_status": "rejected_nonexact_character_residual",
        "parent": {
            "artifact": str(PARENT_PATH.relative_to(ROOT)),
            "letters": len(parent_tape),
            "sha256_normalized": parent_hash,
        },
        "operator": {
            "name": "semantic-first event graph with partial-word owner cuts",
            "provenance": "read-only Luna matrix-scene lane; candidate independently reconstructed here",
            "normalized_parent_spans": [list(LEFT_SPAN), list(RIGHT_SPAN)],
            "raw_parent_spans": [[left_start, left_end], [right_start, right_end]],
            "replaced_left": OLD_LEFT,
            "replaced_right": OLD_RIGHT,
            "new_left": NEW_LEFT,
            "new_right": NEW_RIGHT,
        },
        "candidate": {
            "rendered": rendered,
            "letters": len(tape),
            "growth_over_parent": len(tape) - len(parent_tape),
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
        },
        "novelty_preflight": preflight,
        "parallel_lane_obstructions": [
            {
                "lane": "typed-morphology-valency-chart",
                "normalized_parent_tapes": [
                    parent_tape[morphology_left_span[0] : morphology_left_span[1]],
                    parent_tape[morphology_right_span[0] : morphology_right_span[1]],
                ],
                "normalized_spans": [list(morphology_left_span), list(morphology_right_span)],
                "fresh_units": 1824,
                "attempts": 1824,
                "exact_closures": 0,
                "furthest_prefix_letters": 6,
                "reported_frontier": "Nadia has begun building Mara.",
                "reported_residual": "asbegunbuildingmara",
                "candidate_hash": None,
                "provenance": "read-only Luna report; geometry independently checked in this artifact, chart not rerun",
            },
            {
                "lane": "typed-scene-verb-object-chart",
                "normalized_parent_tapes": [
                    parent_tape[typed_left_span[0] : typed_left_span[1]],
                    parent_tape[typed_right_span[0] : typed_right_span[1]],
                ],
                "normalized_spans": [list(typed_left_span), list(typed_right_span)],
                "fresh_structures": 12000,
                "long_attempts": 11863,
                "exact_closures": 0,
                "furthest_prefix_letters": 3,
                "reported_frontier": "Mara guards Mara and Mara guards Mara.",
                "reported_residual": "aguardsmaraandmaraguardsmara",
                "candidate_hash": None,
                "provenance": "read-only Luna report; reflected geometry independently checked in this artifact, chart not rerun",
            },
            {
                "lane": "partial-word-right-lexicalizer",
                "normalized_spans": [[205, 229], [339, 363]],
                "exact_parent_residual": True,
                "reported_attempt_tape": "araaramdrawsatrapstarthedesse",
                "required_reverse_tape": "essedehtratspartaswardmaraara",
                "obstruction_after_forced_suffix": "eht",
                "reason": "would require the ordinary word `the` to continue backward as `eht`, forcing a semordnilap unit",
                "candidate_hash": None,
                "provenance": "read-only Luna report; geometry independently checked in this artifact, lexicalizer not rerun",
            },
        ],
        "admission": {
            "admitted": False,
            "reason": "The inserted tapes differ in length (28 vs 32) and diverge after the forced two-letter `aw` fragment; the left clause also lacks a determiner before `copper cask`.",
            "human_readability_certified": False,
            "reader_evidence": False,
        },
        "next_construction": "Change seam and use suffix-class-first residual typing: preflight the preserved cut words, enumerate a grammatical right suffix before composing its reverse as a left complement, and require equal inserted lengths plus a multi-character compatible prefix before extending either event.",
    }


if __name__ == "__main__":
    payload = build_payload()
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "status": payload["working_status"],
        "letters": payload["candidate"]["letters"],
        "sha256": payload["candidate"]["normalized_sha256"],
        "insert_lengths": [payload["candidate"]["local_equation"]["left_letters"], payload["candidate"]["local_equation"]["right_letters"]],
        "cursor": payload["candidate"]["local_equation"]["matched_prefix_letters"],
        "phrase_preflight": payload["novelty_preflight"]["status"],
    }, indent=2))
