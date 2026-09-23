#!/usr/bin/env python3
"""Jointly author a replacement scene and a second live insertion on pinned 568."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PARENT_PATH = ROOT / "runs/incumbent-560-outer-causal-scene-20261002.json"
OUTPUT_PATH = ROOT / "runs/luna6-reflected-footprint-dual-variable-20260923.json"
PARENT_SHA = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
OLD_REPLACEMENT = "Sara, did I live? Nora, I saw desserts."
LEFT_SCENE = (
    "After rain, Sara followed one lantern through orchard paths and found Nora "
    "beside locked gates; together they carried maps home safely before dawn."
)
RIGHT_SCENE = "At dawn Nora carried maps across the bridge, then returned through rain and told Sara."


def tape(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalpha())


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def outside_in(text: str) -> dict:
    t = tape(text)
    mismatch = next(((i, t[i], t[-1-i]) for i in range(len(t)//2) if t[i] != t[-1-i]), None)
    return {
        "letters": len(t), "exact": mismatch is None,
        "first_mismatch": ({"offset": mismatch[0], "left": mismatch[1], "right": mismatch[2],
                            "right_offset": len(t)-1-mismatch[0]} if mismatch else None),
    }


def project_audit(text: str) -> dict:
    from llm_palindrome.validator import is_palindrome
    return {"exact": bool(is_palindrome(text))}


def main() -> None:
    parent_doc = json.loads(PARENT_PATH.read_text())
    parent = parent_doc["rows"][0]["rendered"]
    parent_tape = tape(parent)
    if sha(parent_tape) != PARENT_SHA or not outside_in(parent)["exact"] or not project_audit(parent)["exact"]:
        raise SystemExit("Pinned 568 parent hash or exactness check failed")
    if parent.count(OLD_REPLACEMENT) != 1 or parent.count("Live on, reviled. Aron, spam.") != 1:
        raise SystemExit("Expected unique phrase preconditions failed")

    replaced = parent.replace(OLD_REPLACEMENT, LEFT_SCENE)
    rendered = replaced.replace(
        "Live on, reviled. Aron, spam.",
        "Live on, reviled. " + RIGHT_SCENE + " Aron, spam.",
    )
    child_tape = tape(rendered)
    left_width = len(tape(LEFT_SCENE))
    old_width = len(tape(OLD_REPLACEMENT))
    right_width = len(tape(RIGHT_SCENE))
    left_start = 204
    left_end = left_start + left_width
    right_parent_cut = 433
    right_start = right_parent_cut + (left_width - old_width)
    right_end = right_start + right_width
    final_n = len(child_tape)
    right_reflection = [final_n - right_end, final_n - right_start]
    joint_start = max(left_start, right_reflection[0])
    joint_end = min(left_end, right_reflection[1])
    joint_positions = list(range(joint_start, joint_end))
    matched_joint = [
        {"left_offset": pos, "left": child_tape[pos],
         "right_offset": final_n - 1 - pos,
         "right": child_tape[final_n - 1 - pos],
         "equal": child_tape[pos] == child_tape[final_n - 1 - pos]}
        for pos in joint_positions
    ]

    audit = outside_in(rendered)
    if audit["exact"]:
        raise SystemExit("Unexpected exact closure; independently re-audit this row")
    left_tokens = [w.lower() for w in re.findall(r"[A-Za-z]+", LEFT_SCENE)]
    right_tokens = [w.lower() for w in re.findall(r"[A-Za-z]+", RIGHT_SCENE)]
    cross_reverse = [[a, b] for a in left_tokens for b in right_tokens if a[::-1] == b]
    all_new_tokens = left_tokens + right_tokens
    self_pal = sorted({w for w in all_new_tokens if w == w[::-1]})
    repeated_within = {
        "left": sorted({w for w in set(left_tokens) if left_tokens.count(w) > 1}),
        "right": sorted({w for w in set(right_tokens) if right_tokens.count(w) > 1}),
    }

    record = {
        "experiment_id": "luna6-reflected-footprint-dual-variable-20260923",
        "status": "rejected_at_retained_to_variable_cursor",
        "method": "jointly author a long replacement scene and an independently authored sentence at the opposite live owner; align their reflection windows before rendering",
        "novelty_preflight": {
            "parent": str(PARENT_PATH.relative_to(ROOT)),
            "parent_sha256": PARENT_SHA,
            "exact_geometry": {"replace_parent_span": [204, 232], "insert_at_parent_cut": 433},
            "preflight_findings": [
                "No tracked run, experiment source, goal record, or novelty-registry entry was found with the exact [204,232) replacement plus cut-433 insertion signature.",
                "Distinct from the 16/543, 64/504, 178/390, 194/204-to-265/269, and 204/232-to-336/352 attempts.",
                "The parent span [204,232) is adjacent to earlier [204,220)/[348,364) and [222,248)/[320,346) probes; this run uses the full sentence span plus a separate outer-right insertion.",
                "The exact authored right-scene phrase has no prior tracked hit in runs, experiments, or the registry.",
            ],
        },
        "provenance": {
            "parent_artifact": str(PARENT_PATH.relative_to(ROOT)),
            "parent_letters": len(parent_tape), "parent_normalized_sha256": sha(parent_tape),
            "parent_outside_in": outside_in(parent), "parent_project_validator": project_audit(parent),
            "replacement_source": OLD_REPLACEMENT, "replacement_parent_span": [204, 232],
            "replacement_text": LEFT_SCENE, "replacement_letters": left_width,
            "opposing_insert_parent_cut": right_parent_cut,
            "opposing_insert_text": RIGHT_SCENE, "opposing_insert_letters": right_width,
            "net_growth": left_width - old_width + right_width,
        },
        "rendered": rendered,
        "candidate": {
            "letters": len(child_tape), "normalized_sha256": sha(child_tape),
            "outside_in": audit, "project_validator": project_audit(rendered), "exact": False,
        },
        "live_ownership": {
            "left_new_scene_span": [left_start, left_end],
            "right_new_scene_span": [right_start, right_end],
            "right_scene_reflection_span": right_reflection,
            "joint_variable_to_variable_span": [joint_start, joint_end],
            "joint_char_obligations": matched_joint,
            "offset_204_is_jointly_owned": any(row["left_offset"] == 204 for row in matched_joint),
            "offset_204_obligation": next((row for row in matched_joint if row["left_offset"] == 204), None),
            "first_whole_tape_mismatch": audit["first_mismatch"],
            "cursor_note": "The two fresh events jointly discharge offset 204 (a=a), but retained parent material still conflicts earlier at offset 135 (d versus the right scene's final a). The next owner handoff must also replace the [135,204) retained prefix or shift the second insertion's reflection window to begin at the left replacement.",
        },
        "boundary_audit": {
            "left_scene_tokens": left_tokens,
            "right_scene_tokens": right_tokens,
            "cross_scene_whole_token_reversal_pairs": cross_reverse,
            "self_palindromic_tokens": self_pal,
            "repeated_tokens_within_each_scene": repeated_within,
            "shortcut_free_new_units": not (cross_reverse or self_pal or repeated_within["left"] or repeated_within["right"]),
            "discourse_referent_reuse": ["Nora", "Sara"],
            "discourse_note": "Nora and Sara recur as linked participants; no clause, reversed token pair, or palindromic word is duplicated.",
        },
        "readability": {
            "status": "not_reader_evaluated",
            "reason": "The exact tape fails before the first whole-tape cursor reaches the newly jointly owned offset 204; no reader claim is made.",
        },
        "next_operator": "Replace the retained prefix [135,204) as well as the [204,232) span, while keeping the cut-433 event live; this makes the current first mismatch at 135 jointly lexicalized. Preflight that exact three-boundary ownership geometry and author one scene realization, rather than swapping either current clause.",
    }
    OUTPUT_PATH.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(OUTPUT_PATH), "letters": len(child_tape), "sha256": sha(child_tape),
        "first_mismatch": audit["first_mismatch"], "joint_span": [joint_start, joint_end],
        "offset_204": record["live_ownership"]["offset_204_obligation"],
    }, indent=2))


if __name__ == "__main__":
    main()
