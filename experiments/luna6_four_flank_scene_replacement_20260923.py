#!/usr/bin/env python3
"""One four-flank scene replacement on the pinned 568-letter parent.

The bounded construction state fixes the four surface joins before prose is
authored: an answer after ``A tub?``, a complete clause before ``Now, Noel``,
a continuation after ``Leon won.``, and an interjection before the retained
``but a star...`` clause. The live character equation is then evaluated over
the two equal-width replacement surfaces. No search bank or reader metric is
used.
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

PARENT_REL = "runs/incumbent-560-outer-causal-scene-20261002.json"
PARENT_SHA = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
LEFT_SPAN = (125, 148)
RIGHT_SPAN = (420, 443)
LEFT_SCENE = "Heavens, Maya found a map, and Ruth took one to Ken before dawn."
RIGHT_SCENE = "Ken sent Ruth to find Maya; she carried the map home at dawn. Eh,"


def normalize(text: str) -> str:
    return "".join(c.lower() for c in text if c.isascii() and c.isalpha())


def raw_letter_cut(text: str, offset: int) -> int:
    """Index of the offset-th normalized letter; end cuts retain following text."""
    seen = 0
    for i, c in enumerate(text):
        if c.isascii() and c.isalpha():
            if seen == offset:
                return i
            seen += 1
    if seen == offset:
        return len(text)
    raise ValueError(f"normalized offset {offset} exceeds source")


def outside_in(tape: str) -> dict[str, Any]:
    i, j, count = 0, len(tape) - 1, 0
    while i < j:
        count += 1
        if tape[i] != tape[j]:
            return {"exact": False, "comparisons": count,
                    "first_mismatch": {"left_offset": i, "left": tape[i],
                                       "right_offset": j, "right": tape[j]}}
        i += 1
        j -= 1
    return {"exact": True, "comparisons": count, "first_mismatch": None}


def preflight() -> dict[str, Any]:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()
    query = ["git", "grep", "-n", "-F", "-e", "[125,148)", "-e", "[420,443)",
             head, "--", "runs", "experiments", "docs"]
    result = subprocess.run(query, cwd=ROOT, check=False, capture_output=True, text=True)
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr)
    phrase_result = subprocess.run(
        ["git", "grep", "-n", "-F", "-i", "-e", LEFT_SCENE, "-e", RIGHT_SCENE,
         head, "--", "runs", "experiments", "docs", "data"],
        cwd=ROOT, check=False, capture_output=True, text=True)
    if phrase_result.returncode not in (0, 1):
        raise RuntimeError(phrase_result.stderr)
    return {
        "revision": head,
        "exact_geometry_hits": result.stdout.splitlines(),
        "exact_geometry_status": "clear" if not result.stdout else "collision",
        "authored_phrase_hits": phrase_result.stdout.splitlines(),
        "authored_phrase_status": "clear" if not phrase_result.stdout else "collision",
        "logged_2026_09_23_geometries_excluded": [
            "[7,232)+cut561", "[91,232)+cut477", "[135,232)+cut433",
            "[204,232)+cut433", "[204,232)+delete[336,352)",
            "replace[194,204)+delete[265,269)", "insert at reflected sentence cuts [108,460)",
        ],
        "operator_state_new_beyond_coordinates": (
            "typed four-flank scene realization: answer slot after A tub?; complete left clause before Now, Noel; "
            "new right clause after Leon won; right surface ends with Eh, to join the retained but-clause. "
            "All four joins are fixed before lexicalization and carried with one equal-width live character obligation."
        ),
        "known_operator_collisions": [
            "Generic reflected-span replacement has historical precedents; this experiment claims novelty only for "
            "the bounded four-flank attachment state plus exact [125,148)/[420,443) footprint."
        ],
    }


def token_audit(left: str, right: str) -> dict[str, Any]:
    lt = [normalize(x) for x in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", left)]
    rt = [normalize(x) for x in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", right)]
    reverse_pairs = [{"left": a, "right": b} for a in lt for b in rt
                     if len(a) > 1 and a == b[::-1]]
    return {
        "left_tokens": lt, "right_tokens": rt,
        "whole_token_sequence_mirror": lt == list(reversed(rt)),
        "cross_surface_reverse_token_pairs": reverse_pairs,
        "self_palindromic_multi_letter_tokens": sorted({w for w in lt + rt
                                                         if len(w) > 1 and w == w[::-1]}),
        "repeated_tokens_left": sorted({w for w in lt if lt.count(w) > 1}),
        "repeated_tokens_right": sorted({w for w in rt if rt.count(w) > 1}),
        "note": "A non-exact diagnostic cannot be admitted even when these simple shortcut checks are clean.",
    }


def build() -> dict[str, Any]:
    source = json.loads((ROOT / PARENT_REL).read_text())
    parent = source["rows"][0]["rendered"]
    tape = normalize(parent)
    sha = hashlib.sha256(tape.encode("ascii")).hexdigest()
    if len(tape) != 568 or sha != PARENT_SHA:
        raise AssertionError("pinned 568 parent identity changed")
    if RIGHT_SPAN != (568 - LEFT_SPAN[1], 568 - LEFT_SPAN[0]):
        raise AssertionError("selected spans are not reflected")
    ltxt, rtxt = LEFT_SCENE, RIGHT_SCENE
    lnorm, rnorm = normalize(ltxt), normalize(rtxt)
    if len(lnorm) != len(rnorm):
        raise AssertionError("reflected replacement surfaces must have equal letter lengths")

    l0, l1 = raw_letter_cut(parent, LEFT_SPAN[0]), raw_letter_cut(parent, LEFT_SPAN[1])
    r0, r1 = raw_letter_cut(parent, RIGHT_SPAN[0]), raw_letter_cut(parent, RIGHT_SPAN[1])
    if normalize(parent[:l0]) != tape[:LEFT_SPAN[0]] or normalize(parent[l0:l1]) != tape[slice(*LEFT_SPAN)]:
        raise AssertionError("left raw span mapping failed")
    if normalize(parent[:r0]) != tape[:RIGHT_SPAN[0]] or normalize(parent[r0:r1]) != tape[slice(*RIGHT_SPAN)]:
        raise AssertionError("right raw span mapping failed")

    # These raw slices include the original punctuation/space immediately
    # before each following token; keep replacement punctuation at the join.
    candidate = parent[:l0] + ltxt + " " + parent[l1:r0] + rtxt + " " + parent[r1:]
    child = normalize(candidate)
    ptr = outside_in(child)
    fwd = hashlib.sha256(child.encode("ascii")).hexdigest()
    rev = hashlib.sha256(child[::-1].encode("ascii")).hexdigest()
    project_exact = bool(project_validator(candidate))

    cursor = 0
    while cursor < min(len(lnorm), len(rnorm)) and lnorm[cursor] == rnorm[::-1][cursor]:
        cursor += 1
    local = {
        "left_surface": ltxt, "right_surface": rtxt,
        "left_letters": len(lnorm), "right_letters": len(rnorm),
        "left_tape": lnorm, "right_reverse_obligation": rnorm[::-1],
        "matched_prefix": lnorm[:cursor], "first_mismatch_cursor": cursor,
        "left_emits": lnorm[cursor:cursor + 1], "right_requires": rnorm[::-1][cursor:cursor + 1],
        "left_remaining": len(lnorm) - cursor, "right_remaining": len(rnorm) - cursor,
        "closed": cursor == len(lnorm) == len(rnorm),
    }
    return {
        "experiment_id": "luna6-four-flank-scene-replacement-20260923",
        "status": "rejected_live_residual" if not ptr["exact"] else "exact_requires_reader_evidence",
        "method": "single reflected scene replacement with pretyped four-flank syntactic joins and a live equal-width residual",
        "parent": {"artifact": PARENT_REL, "letters": len(tape), "normalized_sha256": sha},
        "novelty_preflight": preflight(),
        "edit": {"left_parent_span": list(LEFT_SPAN), "right_parent_span": list(RIGHT_SPAN),
                 "left_parent_text": parent[l0:l1], "right_parent_text": parent[r0:r1],
                 "left_new_text": ltxt, "right_new_text": rtxt,
                 "left_output_span": [LEFT_SPAN[0], LEFT_SPAN[0] + len(lnorm)],
                 "right_output_span": [RIGHT_SPAN[0] + len(lnorm) - (LEFT_SPAN[1]-LEFT_SPAN[0]),
                                       RIGHT_SPAN[0] + 2*len(lnorm) - (LEFT_SPAN[1]-LEFT_SPAN[0])]},
        "four_flank_state": {
            "left_before": "A tub? ", "left_after": "Now, Noel, did I live?",
            "right_before": "Leon won. ", "right_after": ", but a star spots Aram.",
            "join_requirements": ["answer-like complete left opening", "left insertion ends before a new Now clause",
                                  "right insertion follows a finite Leon clause", "right insertion ends in Eh to form Eh, but..."],
        },
        "rendered": candidate,
        "letters": len(child), "growth_over_parent": len(child) - 568,
        "live_residual": local,
        "exact_audit": {"independent_outside_in": ptr, "project_validator_exact": project_exact,
                        "sha256_forward": fwd, "sha256_reverse": rev,
                        "hashes_equal": fwd == rev, "normalized_sha256": fwd},
        "shortcut_boundary_audit": token_audit(ltxt, rtxt),
        "readability": {"human_evidence": False, "reader_eligible": False,
                        "reason": "The full tape fails exact validation."},
        "failure_and_one_pivot": {
            "obstruction": local,
            "next_operator": "Change the boundary attachment topology once: move the right retained comma/but clause into the authored right event and shift the reflected right boundary to the end of `Aram.`; this removes the forced `Eh` terminal and gives the outer right lexical owner a variable final word. Preflight the new reflected cuts before any further lexicalization.",
        },
    }


if __name__ == "__main__":
    result = build()
    out = ROOT / "runs" / "luna6-four-flank-scene-replacement-20260923.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": result["status"], "letters": result["letters"],
                      "growth": result["growth_over_parent"],
                      "first_mismatch": result["exact_audit"]["independent_outside_in"]["first_mismatch"],
                      "local_residual": result["live_residual"]}, ensure_ascii=False))
