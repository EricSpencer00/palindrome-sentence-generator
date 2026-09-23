#!/usr/bin/env python3
"""Independently audit an exact but wordwise-shortcut 574-letter splice."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.validator import is_palindrome as project_is_palindrome

PARENT_PATH = ROOT / "runs" / "incumbent-560-outer-causal-scene-20261002.json"
OUT_PATH = ROOT / "runs" / "audit-incumbent-568-574-crossed-seam-shortcut-20260923.json"
PARENT_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
PREFLIGHT_REVISION = "b23c659c"
OLD_LEFT = "Noel, I sit. Pat notes. Mara saw God"
OLD_RIGHT = "Dog was Aram. Seton, tap. 'Tis I, Leon"
NEW_LEFT = "Diana was raw as a nomad draws a trap"
NEW_RIGHT = "Part a sward, Damon. Asa, war saw Anaid"
LEFT_SPAN = (178, 204)
RIGHT_SPAN = (364, 390)
EXPECTED_SHA256 = "4c2d2eec9a2bcf0490d760bbe411c58e4f40f323e7e373ddda31dd4800d9d25e"


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


def word_spans(text: str) -> list[tuple[str, int, int]]:
    result = []
    cursor = 0
    for match in re.finditer(r"[A-Za-z]+(?:'[A-Za-z]+)?", text):
        token = normalize(match.group())
        result.append((token, cursor, cursor + len(token)))
        cursor += len(token)
    return result


def literal_preflight(phrases: tuple[str, ...]) -> dict[str, Any]:
    command = ["git", "grep", "-n", "-F", "-i"]
    for phrase in phrases:
        command.extend(("-e", phrase))
    command.extend((PREFLIGHT_REVISION, "--", "runs", "experiments", "docs", "data"))
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if result.returncode not in (0, 1):
        raise RuntimeError(f"literal phrase preflight failed: {result.stderr}")
    hits = result.stdout.splitlines()
    return {
        "revision": PREFLIGHT_REVISION,
        "phrases": list(phrases),
        "scope": "tracked runs/, experiments/, docs/, and data/",
        "hits": hits,
        "status": "no_literal_hits" if not hits else "literal_hits",
    }


def build_payload() -> dict[str, Any]:
    source = json.loads(PARENT_PATH.read_text())
    parent_row = next(row for row in source["rows"] if row["working_status"] == "working_length_incumbent")
    parent = str(parent_row["rendered"])
    parent_tape = normalize(parent)
    parent_hash = hashlib.sha256(parent_tape.encode("ascii")).hexdigest()
    if len(parent_tape) != 568 or parent_hash != PARENT_SHA256:
        raise AssertionError("pinned 568 parent changed")
    if parent_tape[LEFT_SPAN[0] : LEFT_SPAN[1]] != normalize(OLD_LEFT):
        raise AssertionError("left raw source does not match its parent-tape span")
    if parent_tape[RIGHT_SPAN[0] : RIGHT_SPAN[1]] != normalize(OLD_RIGHT):
        raise AssertionError("right raw source does not match its parent-tape span")
    if parent_tape[LEFT_SPAN[0] : LEFT_SPAN[1]] != parent_tape[RIGHT_SPAN[0] : RIGHT_SPAN[1]][::-1]:
        raise AssertionError("selected parent spans are not a reflected pair")

    left_start = parent.index(OLD_LEFT)
    left_end = left_start + len(OLD_LEFT)
    right_start = parent.index(OLD_RIGHT)
    right_end = right_start + len(OLD_RIGHT)
    rendered = parent[:left_start] + NEW_LEFT + parent[left_end:right_start] + NEW_RIGHT + parent[right_end:]
    candidate_tape = normalize(rendered)
    left_tape = normalize(NEW_LEFT)
    right_tape = normalize(NEW_RIGHT)
    if left_tape != right_tape[::-1]:
        raise AssertionError("agent-reported local exact equation did not reproduce")

    scan = outside_in(candidate_tape)
    forward = hashlib.sha256(candidate_tape.encode("ascii")).hexdigest()
    reverse = hashlib.sha256(candidate_tape[::-1].encode("ascii")).hexdigest()
    project_exact = project_is_palindrome(rendered)
    if not scan["exact"] or not project_exact or forward != reverse:
        raise AssertionError("exact candidate did not pass all independent checkers")
    if forward != EXPECTED_SHA256:
        raise AssertionError("candidate hash drifted from the lane report")

    left_words = word_spans(NEW_LEFT)
    right_words = word_spans(NEW_RIGHT)
    left_boundaries = {end for _, _, end in left_words[:-1]}
    right_boundaries = {len(right_tape) - end for _, _, end in right_words[:-1]}
    shared_boundaries = sorted(left_boundaries & right_boundaries)
    boundary_coverage = len(shared_boundaries) / max(1, len(right_boundaries))
    semantic_mirror_pairs = [
        {"left": "Diana", "right": "Anaid", "relation": "exact spelling reversal"},
        {"left": "was", "right": "saw", "relation": "exact spelling reversal"},
        {"left": "raw", "right": "war", "relation": "exact spelling reversal"},
        {"left": "nomad", "right": "Damon", "relation": "exact spelling reversal"},
        {"left": "draws", "right": "sward", "relation": "exact spelling reversal"},
        {"left": "trap", "right": "part", "relation": "exact spelling reversal"},
    ]
    decomposition = []
    union_cuts = sorted({0, len(left_tape), *left_boundaries, *right_boundaries})
    for start, end in zip(union_cuts, union_cuts[1:]):
        right_start = len(right_tape) - end
        right_end = len(right_tape) - start
        right_authored_unit = right_tape[right_start:right_end]
        decomposition.append({
            "span": [start, end],
            "left_unit": left_tape[start:end],
            "right_authored_unit": right_authored_unit,
            "left_equals_reverse_of_right_unit": left_tape[start:end] == right_authored_unit[::-1],
            "left_unit_is_whole_token": any(a == start and b == end for _, a, b in left_words),
            "right_unit_is_whole_token": any(
                len(right_tape) - b == start and len(right_tape) - a == end
                for _, a, b in right_words
            ),
            "self_palindromic_unit": left_tape[start:end] == left_tape[start:end][::-1],
        })
    phrase_audit = literal_preflight((
        NEW_LEFT,
        NEW_RIGHT,
        "Diana was raw as a nomad draws a trap",
        "Part a sward, Damon",
        "Asa, war saw Anaid",
    ))

    return {
        "experiment_id": "audit-incumbent-568-574-crossed-seam-shortcut-20260923",
        "working_status": "exact_but_rejected_near_wordwise_mirror_and_unreadable_draft",
        "parent": {
            "artifact": str(PARENT_PATH.relative_to(ROOT)),
            "letters": len(parent_tape),
            "sha256_normalized": parent_hash,
        },
        "provenance": "newly proposed by a read-only Luna quoted-pair lane; child independently reconstructed and audited here",
        "edit": {
            "normalized_parent_spans": [list(LEFT_SPAN), list(RIGHT_SPAN)],
            "raw_parent_spans": [[left_start, left_end], [right_start, right_end]],
            "replaced_left": OLD_LEFT,
            "replaced_right": OLD_RIGHT,
            "new_left": NEW_LEFT,
            "new_right": NEW_RIGHT,
            "retained_middle_unchanged": True,
        },
        "candidate": {
            "rendered": rendered,
            "letters": len(candidate_tape),
            "growth_over_parent": len(candidate_tape) - len(parent_tape),
            "normalized_sha256": forward,
            "audit": {
                "independent_outside_in_exact": scan["exact"],
                "outside_in_comparisons": scan["comparisons"],
                "project_validator_exact": project_exact,
                "sha256_forward": forward,
                "sha256_reverse": reverse,
                "hashes_equal": forward == reverse,
            },
            "local_equation": {
                "left_tape": left_tape,
                "right_tape": right_tape,
                "exact": left_tape == right_tape[::-1],
                "letters_each": len(left_tape),
            },
        },
        "anti_shortcut_audit": {
            "left_words": [token for token, _, _ in left_words],
            "right_words": [token for token, _, _ in right_words],
            "whole_token_sequence_is_reverse": [token for token, _, _ in left_words]
            == list(reversed([token for token, _, _ in right_words])),
            "left_internal_boundaries": sorted(left_boundaries),
            "reflected_right_internal_boundaries": sorted(right_boundaries),
            "shared_reflected_boundaries": shared_boundaries,
            "reflected_right_boundary_coverage": boundary_coverage,
            "aligned_semordnilap_word_pairs": semantic_mirror_pairs,
            "exact_local_decomposition": decomposition,
            "decision": "rejected: all reflected right word boundaries land on left word boundaries (coverage 1.0), six obvious whole-word spelling reversals occur, and the remaining `Asa` maps to `as` + `a`, including self-palindromic units.",
        },
        "novelty_preflight": phrase_audit,
        "readability_review": {
            "human_reader_evidence": False,
            "certified_readable": False,
            "author_diagnostic": "The left clause is strained (`raw as a nomad draws a trap`); the right imperative/proper-name sequence is not fluent English.",
        },
        "admission": {
            "admitted": False,
            "reason": "Exactness and +6 length do not overcome the near wordwise-reversal shortcut and strained prose.",
        },
        "next_action": "Use a different partial-word parent seam. Before drafting, require complementary word-boundary masks and exclude semordnilap/name-reversal token mappings; then lexicalize a connected scene under the live character residual.",
    }


if __name__ == "__main__":
    payload = build_payload()
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "status": payload["working_status"],
        "letters": payload["candidate"]["letters"],
        "sha256": payload["candidate"]["normalized_sha256"],
        "reflected_boundary_coverage": payload["anti_shortcut_audit"]["reflected_right_boundary_coverage"],
        "semordnilap_pairs": len(payload["anti_shortcut_audit"]["aligned_semordnilap_word_pairs"]),
        "novelty_status": payload["novelty_preflight"]["status"],
    }, indent=2))
