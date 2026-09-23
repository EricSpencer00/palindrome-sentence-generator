#!/usr/bin/env python3
"""One connected-scene attempt on the preserved 560-letter frontier.

The selected owner windows are the imperative/question chain and assertion/
imperative chain at [131,159)/[401,429).  The new clauses describe one chart
handoff scene and deliberately do not reuse the source's mirrored tokens.
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
from llm_palindrome.validator import is_palindrome as project_validator

PARENT_REL = "runs/incumbent-550-central-event-bridge-20261002.json"
PARENT_SHA = "b5f98bfb0b44b31d8cbf78727672a74b588980e1fc8f1ff522a2c4ad1d800ccc"
LEFT_SPAN = (131, 159)
RIGHT_SPAN = (401, 429)
LEFT_SCENE = "At a distant pier, Maya found a tide chart and called Ken."
RIGHT_SCENE = "Ken checked every chart and later logged all tide data."


def norm(text: str) -> str:
    return "".join(c.lower() for c in text if c.isascii() and c.isalpha())


def raw_cut(text: str, offset: int) -> int:
    seen = 0
    for i, char in enumerate(text):
        if char.isascii() and char.isalpha():
            if seen == offset:
                return i
            seen += 1
    if seen == offset:
        return len(text)
    raise ValueError(offset)


def outside_in(tape: str) -> dict[str, Any]:
    i, j, comparisons = 0, len(tape) - 1, 0
    while i < j:
        comparisons += 1
        if tape[i] != tape[j]:
            return {"exact": False, "comparisons": comparisons,
                    "first_mismatch": {"left_offset": i, "left": tape[i],
                                       "right_offset": j, "right": tape[j]}}
        i += 1
        j -= 1
    return {"exact": True, "comparisons": comparisons, "first_mismatch": None}


def token_audit(left: str, right: str) -> dict[str, Any]:
    lt = [norm(w) for w in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", left)]
    rt = [norm(w) for w in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", right)]
    reverse_pairs = [{"left": a, "right": b} for a in lt for b in rt
                     if len(a) > 1 and a == b[::-1]]
    return {
        "left_tokens": lt,
        "right_tokens": rt,
        "whole_token_sequence_mirror": lt == list(reversed(rt)),
        "cross_surface_reverse_token_pairs": reverse_pairs,
        "self_palindromic_multi_letter_tokens": sorted(
            {w for w in lt + rt if len(w) > 1 and w == w[::-1]}),
        "repeated_tokens_left": sorted({w for w in lt if lt.count(w) > 1}),
        "repeated_tokens_right": sorted({w for w in rt if rt.count(w) > 1}),
    }


def preflight() -> dict[str, Any]:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()
    signature = "[131,159)/[401,429) on 560-letter central-event frontier"
    # Search the pinned revision, not the mutable worktree: otherwise this
    # script and its own run artifact become false-positive novelty hits on a
    # rerun.
    geometry = subprocess.run(["git", "grep", "-n", "-F", signature, head,
                               "--", "docs", "runs", "experiments"], cwd=ROOT,
                              capture_output=True, text=True)
    phrase = subprocess.run(["git", "grep", "-n", "-F", "-i", "-e", LEFT_SCENE,
                             "-e", RIGHT_SCENE, head, "--", "runs", "experiments",
                             "docs", "data"], cwd=ROOT, capture_output=True, text=True)
    if geometry.returncode not in (0, 1) or phrase.returncode not in (0, 1):
        raise RuntimeError(geometry.stderr + phrase.stderr)
    return {
        "revision": head,
        "frontier_artifact": PARENT_REL,
        "exact_geometry_signature": signature,
        "geometry_hits": geometry.stdout.splitlines(),
        "geometry_status": "clear" if not geometry.stdout else "collision",
        "authored_phrase_hits_at_head": phrase.stdout.splitlines(),
        "phrase_status": "clear" if not phrase.stdout else "collision",
        "scope_note": "Frontier-specific geometry is unlogged; the old source clauses also occur in the 568 lineage, so novelty is the fresh connected scene and discourse-owner realization, not the raw words or general replacement class.",
    }


def build() -> dict[str, Any]:
    source = json.loads((ROOT / PARENT_REL).read_text())
    parent = source["rows"][0]["rendered"]
    parent_tape = norm(parent)
    parent_sha = hashlib.sha256(parent_tape.encode("ascii")).hexdigest()
    if len(parent_tape) != 560 or parent_sha != PARENT_SHA:
        raise AssertionError("preserved 560 frontier changed")
    if RIGHT_SPAN != (560 - LEFT_SPAN[1], 560 - LEFT_SPAN[0]):
        raise AssertionError("replacement windows are not reflected")

    left_tape, right_tape = norm(LEFT_SCENE), norm(RIGHT_SCENE)
    if len(left_tape) != len(right_tape):
        raise AssertionError(f"equal insertion lengths required: {len(left_tape)} != {len(right_tape)}")
    l0, l1 = raw_cut(parent, LEFT_SPAN[0]), raw_cut(parent, LEFT_SPAN[1])
    r0, r1 = raw_cut(parent, RIGHT_SPAN[0]), raw_cut(parent, RIGHT_SPAN[1])
    if norm(parent[l0:l1]) != parent_tape[slice(*LEFT_SPAN)]:
        raise AssertionError("left normalized span mismatch")
    if norm(parent[r0:r1]) != parent_tape[slice(*RIGHT_SPAN)]:
        raise AssertionError("right normalized span mismatch")

    rendered = parent[:l0] + LEFT_SCENE + " " + parent[l1:r0] + RIGHT_SCENE + " " + parent[r1:]
    child = norm(rendered)
    local_cursor = 0
    right_obligation = right_tape[::-1]
    while local_cursor < min(len(left_tape), len(right_obligation)) and left_tape[local_cursor] == right_obligation[local_cursor]:
        local_cursor += 1
    audit = outside_in(child)
    forward_sha = hashlib.sha256(child.encode("ascii")).hexdigest()
    reverse_sha = hashlib.sha256(child[::-1].encode("ascii")).hexdigest()
    exact_by_project = bool(project_validator(rendered))
    token_boundary_audit = token_audit(LEFT_SCENE, RIGHT_SCENE)

    raw_left = parent[l0:l1]
    raw_right = parent[r0:r1]
    return {
        "experiment_id": "luna6-frontier560-discourse-owner-reanalysis-20260923",
        "status": "exact_requires_readers" if audit["exact"] else "rejected_live_residual",
        "method": "one connected tide-chart handoff scene composed under an equal-width reflected residual; reassign the old imperative/question and assertion/imperative owner chains without reusing their mirrored tokens",
        "parent": {"artifact": PARENT_REL, "letters": 560, "normalized_sha256": parent_sha,
                   "working_frontier_only": True, "568_incumbent_untouched": True},
        "novelty_preflight": preflight(),
        "edit": {"parent_spans": [list(LEFT_SPAN), list(RIGHT_SPAN)],
                 "left_old_text": raw_left, "right_old_text": raw_right,
                 "left_old_tape": parent_tape[slice(*LEFT_SPAN)],
                 "right_old_tape": parent_tape[slice(*RIGHT_SPAN)],
                 "old_local_equation_exact": parent_tape[slice(*LEFT_SPAN)] == parent_tape[slice(*RIGHT_SPAN)][::-1],
                 "left_new_text": LEFT_SCENE, "right_new_text": RIGHT_SCENE,
                 "left_new_tape": left_tape, "right_new_tape": right_tape,
                 "new_local_equation_exact": left_tape == right_obligation},
        "four_flank_joins": {
            "left_before": "He maps Nora.",
            "left_join": "He maps Nora. At a distant pier, Maya found a tide chart and called Ken. Nora saw Noel live.",
            "left_after": "Nora saw Noel live.",
            "right_before": "\u201cEvil Leon\u201d was Aron.",
            "right_join": "\u201cEvil Leon\u201d was Aron. Ken checked every chart and later logged all tide data. Aron, spam.",
            "right_after": "Aron, spam.",
            "assessment": "All four joins are complete sentence boundaries. The new clauses form a linked chart/data event, but retained surrounding discourse is not claimed as globally coherent prose."
        },
        "rendered": rendered,
        "length": len(child), "growth_over_parent": len(child) - 560,
        "live_residual": {"left_tape": left_tape, "right_reverse_obligation": right_obligation,
                          "matched_prefix": left_tape[:local_cursor], "cursor": local_cursor,
                          "left_emits": left_tape[local_cursor:local_cursor+1],
                          "right_requires": right_obligation[local_cursor:local_cursor+1],
                          "remaining": len(left_tape) - local_cursor,
                          "first_obstruction": "at the next character after the shared prefix" if local_cursor < len(left_tape) else None},
        "independent_audit": {
            "normalized_ascii_letters": len(child),
            "outside_in_two_pointer": audit,
            "project_validator_exact": exact_by_project,
            "sha256_forward": forward_sha,
            "sha256_reverse": reverse_sha,
            "sha_equal": forward_sha == reverse_sha,
            "normalized_candidate_sha256": forward_sha,
            "normalization": "lowercase ASCII letters only",
        },
        "shortcut_and_boundary_audit": token_boundary_audit,
        "provenance": {"source": PARENT_REL, "parent_normalized_sha256": parent_sha,
                       "new_content_authored_for_this_run": True,
                       "catalogue_text_imported": False,
                       "generated_text": [LEFT_SCENE, RIGHT_SCENE]},
        "readability": {"human_reader_evidence": False,
                        "claim": "The linked clauses are readable English in isolation; the complete rendered child has not been reader-tested."},
        "next_operator": "After preserving this cursor-specific obstruction, pivot to a different seam/operator on the 558 or 556 frontier; do not synonym-swap this clause pair or demote the 568 incumbent.",
    }


if __name__ == "__main__":
    result = build()
    target = ROOT / "runs" / "luna6-frontier560-discourse-owner-reanalysis-20260923.json"
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": result["status"], "length": result["length"],
                      "growth": result["growth_over_parent"],
                      "cursor": result["live_residual"]["cursor"],
                      "first_mismatch": result["independent_audit"]["outside_in_two_pointer"]["first_mismatch"],
                      "project_validator_exact": result["independent_audit"]["project_validator_exact"]}, ensure_ascii=False))
