#!/usr/bin/env python3
"""Reconstruct a context-trapped 568 seam and retain parallel obstructions."""
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
OUT_PATH = ROOT / "runs" / "audit-incumbent-568-context-residual-20260923.json"
PARENT_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
PREFLIGHT_REVISION = "1b34fdb6"
LEFT_RAW = (65, 79)
RIGHT_RAW = (703, 719)
OLD_LEFT = "ra delivers ma"
OLD_RIGHT = "am's reviled, Ar"
NEW_LEFT = "ra brings fresh records; the courier checks the old gri"
NEW_RIGHT = "am's auditor weighs brass; our scribe names the sour bar"
EXPECTED_CANDIDATE_SHA256 = "7dc2b275bca05aa6027ded1834f8f2264edb0ea1abd9cc326963f3983f0e9965"


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
        raise RuntimeError(f"phrase preflight failed: {result.stderr}")
    hits = result.stdout.splitlines()
    return {
        "revision": PREFLIGHT_REVISION,
        "phrases": list(phrases),
        "scope": "tracked runs/, experiments/, docs/, and data/",
        "hits": hits,
        "status": "no_literal_hits" if not hits else "phrase_family_collision",
    }


def geometry_for(tape: str, left: tuple[int, int], right: tuple[int, int]) -> dict[str, Any]:
    left_tape, right_tape = tape[left[0] : left[1]], tape[right[0] : right[1]]
    if left_tape != right_tape[::-1]:
        raise AssertionError(f"source spans are not reflected: {left}, {right}")
    return {"normalized_spans": [list(left), list(right)], "source_tapes": [left_tape, right_tape]}


def build_payload() -> dict[str, Any]:
    source = json.loads(PARENT_PATH.read_text())
    row = next(item for item in source["rows"] if item["working_status"] == "working_length_incumbent")
    parent = str(row["rendered"])
    parent_tape = normalize(parent)
    parent_sha = hashlib.sha256(parent_tape.encode("ascii")).hexdigest()
    if len(parent_tape) != 568 or parent_sha != PARENT_SHA256:
        raise AssertionError("pinned 568 parent changed")
    if parent[LEFT_RAW[0] : LEFT_RAW[1]] != OLD_LEFT or parent[RIGHT_RAW[0] : RIGHT_RAW[1]] != OLD_RIGHT:
        raise AssertionError("raw source spans differ from the reported seam")
    left_norm_span = (50, 62)
    right_norm_span = (506, 518)
    if parent_tape[left_norm_span[0] : left_norm_span[1]] != normalize(OLD_LEFT):
        raise AssertionError("left normalized seam does not match the parent")
    if parent_tape[right_norm_span[0] : right_norm_span[1]] != normalize(OLD_RIGHT):
        raise AssertionError("right normalized seam does not match the parent")
    if parent_tape[left_norm_span[0] : left_norm_span[1]] != parent_tape[right_norm_span[0] : right_norm_span[1]][::-1]:
        raise AssertionError("edited parent fragments are not reflected")

    rendered = parent[: LEFT_RAW[0]] + NEW_LEFT + parent[LEFT_RAW[1] : RIGHT_RAW[0]] + NEW_RIGHT + parent[RIGHT_RAW[1] :]
    candidate_tape = normalize(rendered)
    left_tape, right_tape = normalize(NEW_LEFT), normalize(NEW_RIGHT)
    required = right_tape[::-1]
    cursor = 0
    while cursor < min(len(left_tape), len(required)) and left_tape[cursor] == required[cursor]:
        cursor += 1
    scan = outside_in(candidate_tape)
    candidate_sha = hashlib.sha256(candidate_tape.encode("ascii")).hexdigest()
    reverse_sha = hashlib.sha256(candidate_tape[::-1].encode("ascii")).hexdigest()
    project_exact = project_is_palindrome(rendered)
    if candidate_sha != EXPECTED_CANDIDATE_SHA256:
        raise AssertionError("reconstructed diagnostic differs from reported candidate hash")
    if scan["exact"] or project_exact or candidate_sha == reverse_sha:
        raise AssertionError("reported diagnostic unexpectedly closes")

    typed_geometry = geometry_for(parent_tape, (270, 280), (288, 298))
    quoted_geometry = geometry_for(parent_tape, (259, 269), (299, 309))
    preflight = phrase_preflight((
        "brings fresh records",
        "courier checks the old",
        "auditor weighs brass",
        "scribe names the sour",
    ))
    return {
        "experiment_id": "audit-incumbent-568-context-residual-20260923",
        "working_status": "rejected_nonexact_context_trapped_residual",
        "parent": {
            "artifact": str(PARENT_PATH.relative_to(ROOT)),
            "letters": len(parent_tape),
            "sha256_normalized": parent_sha,
        },
        "operator": {
            "name": "context-aware suffix-first lexicalization diagnostic",
            "provenance": "read-only Luna matrix-scene lane; child and checks independently reconstructed here",
            "preflight_revision": PREFLIGHT_REVISION,
            "normalized_spans": [list(left_norm_span), list(right_norm_span)],
            "raw_spans": [list(LEFT_RAW), list(RIGHT_RAW)],
            "source_fragments": [OLD_LEFT, OLD_RIGHT],
            "insertions": [NEW_LEFT, NEW_RIGHT],
            "retained_context_traps": [
                "left suffix `ps` retains the maps word family",
                "right prefix `Sp` and suffix `'s reviled` retain the Spam's reviled family",
            ],
        },
        "candidate": {
            "rendered": rendered,
            "letters": len(candidate_tape),
            "growth_over_parent": len(candidate_tape) - len(parent_tape),
            "normalized_sha256": candidate_sha,
            "audit": {
                "independent_outside_in_exact": scan["exact"],
                "first_mismatch": scan["first_mismatch"],
                "project_validator_exact": project_exact,
                "sha256_forward": candidate_sha,
                "sha256_reverse": reverse_sha,
                "hashes_equal": candidate_sha == reverse_sha,
            },
            "local_equation": {
                "left_tape": left_tape,
                "right_tape": right_tape,
                "required_reverse_right_tape": required,
                "left_letters": len(left_tape),
                "right_letters": len(right_tape),
                "matched_prefix_letters": cursor,
                "first_mismatch": {"cursor": cursor, "left": left_tape[cursor], "required": required[cursor]},
            },
        },
        "novelty_preflight": preflight,
        "parallel_lane_obstructions": [
            {
                "lane": "typed-valency-aspect-chart",
                **typed_geometry,
                "fresh_units": 16632,
                "attempts": 16632,
                "exact_closures": 0,
                "furthest_prefix_letters": 6,
                "reported_frontier": "Nadia will carry water.",
                "reported_residual": "illcarrywater",
                "candidate_hash": None,
                "provenance": "read-only Luna report; seam geometry independently checked here; chart not rerun",
            },
            {
                "lane": "quoted-attribution-dual-parse",
                **quoted_geometry,
                "forced_left_prefix": "rsi",
                "required_right_suffix": "isr",
                "retained_continuation": "is reviled",
                "reason": "the unchanged right continuation forces the excluded `reviled` phrase family; no candidate was emitted",
                "candidate_hash": None,
                "provenance": "read-only Luna report; seam geometry independently checked here; no bounded chart was run",
            },
        ],
        "admission": {
            "admitted": False,
            "reason": "The local equation matches four letters then diverges (i/u) at the global offset 54; unchanged context also forces the old maps and Spam's-reviled families.",
            "human_readability_certified": False,
            "reader_evidence": False,
        },
        "next_construction": "Do boundary-context admissibility before any lexical search: widen or relocate the cut until both retained edges can form fresh words and clauses, then use a new grammar/operator on a different reflected seam. Reject a cut immediately if its unchanged boundary forces an excluded lexical family.",
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
