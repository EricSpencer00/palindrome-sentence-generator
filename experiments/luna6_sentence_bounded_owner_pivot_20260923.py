#!/usr/bin/env python3
"""One owner-topology pivot after the four-flank [125,148)/[420,443) failure.

Both complete reflected windows are replaced at once.  This removes the
previous retained ``Eh, but...`` constraint and composes a linked map/bridge
scene under one live, equal-width character residual.  It is one candidate
attempt, not a lexical sweep.
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
LEFT_SPAN = (91, 148)
RIGHT_SPAN = (420, 477)
LEFT_SCENE = "At dawn, Maya found the flood map and told Ken the river had swept the bridge away."
RIGHT_SCENE = "Ruth later found the bridge, and Ken carried the map back to the village for Maya."


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


def audit_tokens(left: str, right: str) -> dict[str, Any]:
    lt = [norm(w) for w in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", left)]
    rt = [norm(w) for w in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", right)]
    reversed_pairs = [{"left": a, "right": b} for a in lt for b in rt
                      if len(a) > 1 and a == b[::-1]]
    return {"left_tokens": lt, "right_tokens": rt,
            "whole_token_sequence_mirror": lt == list(reversed(rt)),
            "cross_surface_reverse_token_pairs": reversed_pairs,
            "self_palindromic_multi_letter_tokens": sorted({w for w in lt + rt if len(w) > 1 and w == w[::-1]}),
            "repeated_tokens_left": sorted({w for w in lt if lt.count(w) > 1}),
            "repeated_tokens_right": sorted({w for w in rt if rt.count(w) > 1})}


def novelty() -> dict[str, Any]:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()
    exact_text = f"[{LEFT_SPAN[0]},{LEFT_SPAN[1]})/[{RIGHT_SPAN[0]},{RIGHT_SPAN[1]})"
    # Search all current tracked and untracked project history for the exact
    # geometry; this is a single signature query, not a general sweep.
    search = subprocess.run(["rg", "-n", "-F", exact_text, "docs", "runs", "experiments"],
                            cwd=ROOT, check=False, capture_output=True, text=True)
    if search.returncode not in (0, 1):
        raise RuntimeError(search.stderr)
    phrase = subprocess.run(["git", "grep", "-n", "-F", "-i", "-e", LEFT_SCENE,
                             "-e", RIGHT_SCENE, head, "--", "runs", "experiments", "docs", "data"],
                            cwd=ROOT, check=False, capture_output=True, text=True)
    if phrase.returncode not in (0, 1):
        raise RuntimeError(phrase.stderr)
    return {"revision": head, "exact_geometry_signature": exact_text,
            "current_tree_geometry_hits": search.stdout.splitlines(),
            "geometry_status": "clear" if not search.stdout else "collision",
            "exact_authored_phrase_hits_at_head": phrase.stdout.splitlines(),
            "phrase_status": "clear" if not phrase.stdout else "collision",
            "distinct_from_9_23_geometries": ["[7,232)+cut561", "[91,232)+cut477",
                "[135,232)+cut433", "[204,232)+cut433", "[204,232)+delete[336,352)",
                "replace[194,204)+delete[265,269)", "[125,148)/[420,443) four-flank attempt"],
            "new_operator_state": "symmetric complete sentence-bounded paragraph owner pair; both sides advance through one live character residual; the awkward retained Eh/but boundary is absorbed rather than repaired by shifted insertion",
            "collision_caveat": "Other historical reflected-window replacements exist; this claim is specific to the exact owner signature and removal of the prior forced flank, not to span replacement as a general class."}


def build() -> dict[str, Any]:
    source = json.loads((ROOT / PARENT_REL).read_text())
    parent = source["rows"][0]["rendered"]
    parent_tape = norm(parent)
    parent_sha = hashlib.sha256(parent_tape.encode("ascii")).hexdigest()
    if len(parent_tape) != 568 or parent_sha != PARENT_SHA:
        raise AssertionError("pinned parent changed")
    if RIGHT_SPAN != (568 - LEFT_SPAN[1], 568 - LEFT_SPAN[0]):
        raise AssertionError("pivot spans are not reflected")
    lt, rt = norm(LEFT_SCENE), norm(RIGHT_SCENE)
    if len(lt) != len(rt):
        raise AssertionError("pivot requires equal reflected insertion widths")
    l0, l1 = raw_cut(parent, LEFT_SPAN[0]), raw_cut(parent, LEFT_SPAN[1])
    r0, r1 = raw_cut(parent, RIGHT_SPAN[0]), raw_cut(parent, RIGHT_SPAN[1])
    if norm(parent[:l0]) != parent_tape[:LEFT_SPAN[0]] or norm(parent[l0:l1]) != parent_tape[slice(*LEFT_SPAN)]:
        raise AssertionError("left raw slice mapping failed")
    if norm(parent[:r0]) != parent_tape[:RIGHT_SPAN[0]] or norm(parent[r0:r1]) != parent_tape[slice(*RIGHT_SPAN)]:
        raise AssertionError("right raw slice mapping failed")
    rendered = parent[:l0] + LEFT_SCENE + " " + parent[l1:r0] + RIGHT_SCENE + " " + parent[r1:]
    child = norm(rendered)
    ptr = outside_in(child)
    forward = hashlib.sha256(child.encode("ascii")).hexdigest()
    reverse = hashlib.sha256(child[::-1].encode("ascii")).hexdigest()
    project_exact = bool(project_validator(rendered))
    reverse_right = rt[::-1]
    cursor = 0
    while cursor < len(lt) and lt[cursor] == reverse_right[cursor]:
        cursor += 1
    return {
        "experiment_id": "luna6-sentence-bounded-owner-pivot-20260923",
        "status": "rejected_live_residual" if not ptr["exact"] else "exact_requires_readers",
        "method": "sentence-bounded reflected owner pair after four-flank obstruction; joint map/bridge/flood narrative under live character debt",
        "parent": {"artifact": PARENT_REL, "letters": 568, "normalized_sha256": parent_sha},
        "novelty_preflight": novelty(),
        "edit": {"parent_spans": [list(LEFT_SPAN), list(RIGHT_SPAN)],
                 "left_old_text": parent[l0:l1], "right_old_text": parent[r0:r1],
                 "left_new_text": LEFT_SCENE, "right_new_text": RIGHT_SCENE,
                 "new_left_output_span": [91, 91 + len(lt)],
                 "new_right_output_span": [420 + len(lt) - 57, 420 + 2*len(lt) - 57]},
        "four_flanks": {"left_before": "A tub? He maps Aron.",
                        "left_after": "Now, Noel, did I live?",
                        "right_before": "Leon won.", "right_after": "Nora, spam.",
                        "status": "both inserted paragraphs are sentence-complete; continuation artifacts stay outside the replaced windows"},
        "rendered": rendered,
        "letters": len(child), "growth_over_parent": len(child) - 568,
        "live_residual": {"left_tape": lt, "right_reverse_obligation": reverse_right,
                          "matched_prefix": lt[:cursor], "cursor": cursor,
                          "left_emits": lt[cursor:cursor + 1],
                          "right_requires": reverse_right[cursor:cursor + 1],
                          "remaining_each": len(lt) - cursor,
                          "exact_local_closure": cursor == len(lt)},
        "exact_audit": {"outside_in": ptr, "project_validator_exact": project_exact,
                        "sha256_forward": forward, "sha256_reverse": reverse,
                        "hashes_equal": forward == reverse, "normalized_sha256": forward},
        "shortcut_boundary_audit": audit_tokens(LEFT_SCENE, RIGHT_SCENE),
        "readability": {"reader_evidence": False, "reader_eligible": False,
                        "reason": "The candidate fails exact validation."},
        "failure_and_next_repair": {"obstruction": {"cursor": cursor,
                "left": lt[cursor:cursor + 1], "required": reverse_right[cursor:cursor + 1]},
            "one_falsifier": "If the first residual remains nonzero after this one sentence-boundary owner pivot, retire this seam; do not phrase-swap or add another cut."},
    }


if __name__ == "__main__":
    result = build()
    target = ROOT / "runs" / "luna6-sentence-bounded-owner-pivot-20260923.json"
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": result["status"], "letters": result["letters"],
                      "growth": result["growth_over_parent"],
                      "first_mismatch": result["exact_audit"]["outside_in"]["first_mismatch"],
                      "local_residual": result["live_residual"]}, ensure_ascii=False))
