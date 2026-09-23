#!/usr/bin/env python3
"""Bounded Luna-6 attachment-topology pivot on the pinned 568 parent.

The requested direct lexical reanalysis was preflighted against existing
partial-word/tape-resegmentation records and retired as duplicate.  This one
replacement hypothesis changes the syntax topology instead: a temporal PP
attaches to a ledger-checking clause on the left, while a source PP attaches
to a courier-protection clause on the right.  It is one authored pair, not a
lexical sweep or a readability claim.
"""
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

PARENT_REL = "runs/incumbent-560-outer-causal-scene-20261002.json"
PARENT_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
LEFT_CUT = 108
RIGHT_CUT = 460  # 568 - LEFT_CUT; both are authored sentence boundaries.
LEFT_INSERTION = "Nadia checks the ledgers after the rain."
RIGHT_INSERTION = "The courier protects the maps from rain."


def normalize(text: str) -> str:
    return "".join(c.lower() for c in text if c.isascii() and c.isalpha())


def outside_in(tape: str) -> dict[str, Any]:
    i, j, count = 0, len(tape) - 1, 0
    mismatch = None
    while i < j:
        count += 1
        if tape[i] != tape[j]:
            mismatch = {"left_cursor": i, "right_cursor": j,
                        "left_char": tape[i], "right_char": tape[j]}
            break
        i += 1
        j -= 1
    return {"exact": mismatch is None, "comparisons": count,
            "first_mismatch": mismatch}


def phrase_preflight() -> dict[str, Any]:
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                         check=True, capture_output=True, text=True).stdout.strip()
    result = subprocess.run(
        ["git", "grep", "-n", "-F", "-i", "-e", LEFT_INSERTION,
         "-e", RIGHT_INSERTION, rev, "--", "runs", "experiments", "docs", "data"],
        cwd=ROOT, check=False, capture_output=True, text=True)
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr)
    return {"revision": rev, "scope": "tracked runs/, experiments/, docs/, data/",
            "phrases": [LEFT_INSERTION, RIGHT_INSERTION],
            "literal_hits": result.stdout.splitlines(),
            "status": "clean" if not result.stdout else "collision"}


def raw_cut(text: str, normalized_offset: int) -> int:
    """Map a normalized offset to the raw start of the following word."""
    count = 0
    for i, char in enumerate(text):
        if char.isascii() and char.isalpha():
            if count == normalized_offset:
                # Include whitespace after the previous punctuation in the prefix.
                return i
            count += 1
    if count == normalized_offset:
        return len(text)
    raise ValueError("normalized cut is outside the source tape")


def token_boundaries(text: str) -> list[int]:
    ends, cursor = [], 0
    for token in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text):
        cursor += len(normalize(token))
        ends.append(cursor)
    return ends[:-1]


def build() -> dict[str, Any]:
    payload = json.loads((ROOT / PARENT_REL).read_text())
    parent = payload["rows"][0]["rendered"]
    parent_tape = normalize(parent)
    parent_sha = hashlib.sha256(parent_tape.encode("ascii")).hexdigest()
    if len(parent_tape) != 568 or parent_sha != PARENT_SHA256:
        raise AssertionError("pinned 568 parent identity changed")
    if len(normalize(LEFT_INSERTION)) != len(normalize(RIGHT_INSERTION)):
        raise AssertionError("bounded test requires equal insertion lengths")
    if LEFT_CUT + RIGHT_CUT != len(parent_tape):
        raise AssertionError("the selected cuts are not reflected")

    left_raw, right_raw = raw_cut(parent, LEFT_CUT), raw_cut(parent, RIGHT_CUT)
    if normalize(parent[:left_raw]) != parent_tape[:LEFT_CUT]:
        raise AssertionError("left cut mapping drifted")
    if normalize(parent[:right_raw]) != parent_tape[:RIGHT_CUT]:
        raise AssertionError("right cut mapping drifted")
    candidate = (parent[:left_raw] + LEFT_INSERTION + " " + parent[left_raw:right_raw]
                 + RIGHT_INSERTION + " " + parent[right_raw:])
    tape = normalize(candidate)
    ptr = outside_in(tape)
    fwd = hashlib.sha256(tape.encode("ascii")).hexdigest()
    rev = hashlib.sha256(tape[::-1].encode("ascii")).hexdigest()
    project_exact = project_is_palindrome(candidate)

    left_tape, right_tape = normalize(LEFT_INSERTION), normalize(RIGHT_INSERTION)
    local_cursor = next((i for i, (a, b) in enumerate(zip(left_tape, right_tape[::-1]))
                         if a != b), min(len(left_tape), len(right_tape)))
    local = {
        "left_letters": len(left_tape), "right_letters": len(right_tape),
        "left_surface": LEFT_INSERTION, "right_surface": RIGHT_INSERTION,
        "left_tape": left_tape, "right_tape_read_from_outer_cursor": right_tape[::-1],
        "matched_prefix": left_tape[:local_cursor],
        "cursor": local_cursor,
        "left_emits": left_tape[local_cursor:local_cursor + 1],
        "right_requires": right_tape[::-1][local_cursor:local_cursor + 1],
        "remaining_left_letters": len(left_tape) - local_cursor,
        "remaining_right_letters": len(right_tape) - local_cursor,
        "closed": left_tape == right_tape[::-1],
    }

    # Read-only novelty checks: the chosen geometry is new as an exact cut pair,
    # but the general word-internal/tape resegmentation operator is not new.
    geometry = f"[{LEFT_CUT},{LEFT_CUT}) / [{RIGHT_CUT},{RIGHT_CUT})"
    rg = subprocess.run(["git", "grep", "-F", "-n", geometry, "HEAD", "--",
                         "docs", "runs", "experiments"], cwd=ROOT,
                        check=False, capture_output=True, text=True)
    geometry_hits = rg.stdout.splitlines() if rg.returncode == 0 else []
    return {
        "experiment_id": "luna6-two-direction-reanalysis-20260923",
        "status": "bounded_attachment_topology_obstruction" if not ptr["exact"] else "exact_candidate_requires_readers",
        "method": "single syntactic-attachment topology pivot after duplicate-preflight of direct tape resegmentation",
        "parent": {"artifact": PARENT_REL, "letters": len(parent_tape), "sha256": parent_sha},
        "rendered_candidate": candidate,
        "letters": len(tape), "growth_over_parent": len(tape) - 568,
        "provenance": {
            "left_event": "Nadia checks the ledgers after the rain",
            "right_event": "the courier protects the maps from rain",
            "attachment_topology": {"left": "temporal PP after the rain attaches to checks the ledgers",
                                    "right": "source PP from rain attaches to protects the maps"},
            "borrowed_text": False, "finished_tape_reversal": False,
            "posthoc_character_repair": False, "search_kind": "one fixed authored pair; no lexical sweep",
        },
        "novelty_preflight": {
            "operator": "direct bidirectional lexical reanalysis / tape resegmentation",
            "operator_status": "retired-as-duplicate: tracked work already includes multiple partial-word lexicalizers, character resegmentations, and phrasewise equation audits",
            "selected_attachment_geometry": geometry,
            "geometry_literal_hits": geometry_hits,
            "authored_phrase_preflight": phrase_preflight(),
            "pivot": "one attachment-topology pair, distinct from changing POS banks or morphology",
        },
        "live_residual": local,
        "boundary_audit": {
            "left_internal_boundaries": token_boundaries(LEFT_INSERTION),
            "reflected_right_internal_boundaries": sorted(
                len(right_tape) - b for b in token_boundaries(RIGHT_INSERTION)),
            "whole_token_sequence_mirror": [normalize(x) for x in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", LEFT_INSERTION)] ==
                list(reversed([normalize(x) for x in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", RIGHT_INSERTION)])),
            "interpretation": "This is a failed equation diagnostic, so no non-tokenwise closure is claimed.",
        },
        "exact_audit": {
            "independent_outside_in_exact": ptr["exact"], "outside_in": ptr,
            "project_validator_exact": project_exact,
            "sha256_forward": fwd, "sha256_reverse": rev, "hashes_equal": fwd == rev,
            "normalized_sha256": fwd,
        },
        "readability": {"human_evidence": False, "reader_eligible": False,
                        "reason": "not exact; no reader test"},
        "failure_and_next_repair": {
            "obstruction": local,
            "next_repair": "Keep the two event roles and attachment heads fixed; jointly replace only the first mismatching lexical heads under the live residual, then rerun both retained-flank syntax checks and exact audits. Do not reopen a generic seam/word sweep.",
        },
    }


if __name__ == "__main__":
    result = build()
    out = ROOT / "runs" / "luna6-two-direction-reanalysis-20260923.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": result["status"], "letters": result["letters"],
                      "growth": result["growth_over_parent"],
                      "local_residual": result["live_residual"],
                      "exact": result["exact_audit"]["independent_outside_in_exact"]}))
