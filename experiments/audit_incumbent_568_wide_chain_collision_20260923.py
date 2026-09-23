#!/usr/bin/env python3
"""Audit and reject an exact wider-seam proposal that reuses old event clauses."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.validator import is_palindrome as project_is_palindrome


PARENT = ROOT / "runs" / "incumbent-560-outer-causal-scene-20261002.json"
OUT = ROOT / "runs" / "audit-incumbent-568-wide-chain-collision-20260923.json"
PARENT_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
PREFLIGHT_REVISION = "8c35aba2"
LEFT_SPAN = (16, 52)
RIGHT_SPAN = (516, 552)
LEFT_SURFACE = (
    "Ira. Ira saw Aidan. Aidan stops Mara. Mara stops Sara. "
    "Sara spots Aras. Nora"
)
RIGHT_SURFACE = (
    "Aron. Sara stops Aras. Aras spots Aram. Aram spots Nadia. "
    "Nadia was Ari. Ari"
)
AUTHORED_CLAUSES = (
    "Ira saw Aidan",
    "Aidan stops Mara",
    "Mara stops Sara",
    "Sara spots Aras",
    "Sara stops Aras",
    "Aras spots Aram",
    "Aram spots Nadia",
    "Nadia was Ari",
    "Ari stops flow now",
)


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


@lru_cache(maxsize=1)
def phrase_collisions() -> dict[str, list[str]]:
    command = ["git", "grep", "-n", "-F", "-i"]
    for phrase in AUTHORED_CLAUSES:
        command.extend(("-e", phrase))
    command.extend((PREFLIGHT_REVISION, "--", "runs", "experiments", "docs", "data"))
    result = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
    if result.returncode not in (0, 1):
        raise RuntimeError(f"novelty search failed: {result.stderr}")

    collisions: dict[str, set[str]] = {phrase: set() for phrase in AUTHORED_CLAUSES}
    for line in result.stdout.splitlines():
        parts = line.split(":", 3)
        if len(parts) != 4:
            continue
        path, content = parts[1], parts[3].lower()
        for phrase in AUTHORED_CLAUSES:
            if phrase.lower() in content:
                collisions[phrase].add(path)
    return {phrase: sorted(paths) for phrase, paths in collisions.items() if paths}


def build_payload() -> dict[str, Any]:
    payload = json.loads(PARENT.read_text())
    parent_row = payload["rows"][0]
    parent = str(parent_row["rendered"])
    tape = normalize(parent)
    parent_sha = hashlib.sha256(tape.encode("ascii")).hexdigest()
    if len(tape) != 568 or parent_sha != PARENT_SHA256:
        raise AssertionError("the pinned 568 parent changed")

    raw_map = [index for index, char in enumerate(parent) if char.isascii() and char.isalpha()]
    left_raw = (raw_map[LEFT_SPAN[0]], raw_map[LEFT_SPAN[1] - 1] + 1)
    right_raw = (raw_map[RIGHT_SPAN[0]], raw_map[RIGHT_SPAN[1] - 1] + 1)
    left_start, left_end = left_raw
    right_start, right_end = right_raw
    if normalize(parent[left_start:left_end]) != tape[slice(*LEFT_SPAN)]:
        raise AssertionError("left raw replacement does not match its normalized span")
    if normalize(parent[right_start:right_end]) != tape[slice(*RIGHT_SPAN)]:
        raise AssertionError("right raw replacement does not match its normalized span")
    if tape[:LEFT_SPAN[0]] != tape[RIGHT_SPAN[1]:][::-1]:
        raise AssertionError("the retained outer obligations do not match")

    local_left = normalize(LEFT_SURFACE)
    local_right = normalize(RIGHT_SURFACE)
    if local_left != local_right[::-1]:
        raise AssertionError("the widened replacement leaves a live residual")
    rendered = (
        parent[:left_start]
        + LEFT_SURFACE
        + parent[left_end:right_start]
        + RIGHT_SURFACE
        + parent[right_end:]
    )
    child_tape = normalize(rendered)
    scan = outside_in(child_tape)
    forward = hashlib.sha256(child_tape.encode("ascii")).hexdigest()
    reverse = hashlib.sha256(child_tape[::-1].encode("ascii")).hexdigest()
    project_exact = project_is_palindrome(rendered)
    if not scan["exact"] or forward != reverse or not project_exact:
        raise AssertionError("independent exact checks disagree")
    collisions = phrase_collisions()
    if not collisions:
        raise AssertionError("expected historic event-phrase collisions were not reproduced")

    left_tokens = [normalize(token) for token in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", LEFT_SURFACE)]
    right_tokens = [normalize(token) for token in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", RIGHT_SURFACE)]
    token_sequence_mirror = left_tokens == list(reversed(right_tokens))
    return {
        "experiment_id": "audit-incumbent-568-wide-chain-collision-20260923",
        "working_status": "exact_but_novelty_rejected_not_promoted",
        "parent": {
            "artifact": str(PARENT.relative_to(ROOT)),
            "letters": len(tape),
            "sha256": parent_sha,
        },
        "proposer": "read-only Luna wide-seam lane",
        "seam": {
            "normalized_spans": [list(LEFT_SPAN), list(RIGHT_SPAN)],
            "raw_spans": [list(left_raw), list(right_raw)],
            "left_surface": LEFT_SURFACE,
            "right_surface": RIGHT_SURFACE,
            "left_letters": len(local_left),
            "right_letters": len(local_right),
            "retained_middle_letters": len(normalize(parent[left_end:right_start])),
        },
        "candidate": {
            "rendered": rendered,
            "letters": len(child_tape),
            "growth_over_parent": len(child_tape) - len(tape),
            "normalized_sha256": forward,
            "local_equation": {"left": local_left, "reverse_of_right": local_right[::-1]},
            "audit": {
                "independent_outside_in_exact": scan["exact"],
                "outside_in_comparisons": scan["comparisons"],
                "first_mismatch": scan["first_mismatch"],
                "project_validator_exact": project_exact,
                "sha256_forward": forward,
                "sha256_reverse": reverse,
                "hashes_equal": forward == reverse,
            },
            "whole_token_sequence_mirror": token_sequence_mirror,
        },
        "novelty_preflight": {
            "revision": PREFLIGHT_REVISION,
            "scope": "case-insensitive literal clause search in tracked runs/, experiments/, docs/, and data/",
            "authored_clauses": list(AUTHORED_CLAUSES),
            "collision_paths_by_clause": collisions,
            "reused_clause_count": len(collisions),
            "status": "failed",
        },
        "decision": "Reject from the working frontier despite exactness and length: multiple event clauses are recycled from prior linked-event experiments.",
        "reader_evidence": False,
        "readability_claim": False,
        "next_action": "Choose a different actual 568 seam and a semantic-first topology; exclude the linked stop/spot clause bank before lexical realization.",
    }


if __name__ == "__main__":
    result = build_payload()
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": result["working_status"], "letters": result["candidate"]["letters"], "sha256": result["candidate"]["normalized_sha256"], "reused_clauses": result["novelty_preflight"]["reused_clause_count"]}, indent=2))
