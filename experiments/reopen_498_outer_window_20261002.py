"""Audit editable mirrored windows on the genuine 498-letter incumbent.

The 498-letter child is exact but its 129-letter outer spans are word-salad.
This probe asks the precise next question: how much of that shell must be
reopened before both cut points land at complete word boundaries?  A new
56-letter discourse shell is used only as an exactness/control fixture.  Rows
that split an inherited word or retain the old filler are not promoted.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "runs" / "overhang-growth-from-240-20261001.json"
SHELL = ROOT / "runs" / "first-person-discourse-abba-20261002.json"
OUT = ROOT / "runs" / "reopen-498-outer-window-20261002.json"
sys.path.insert(0, str(ROOT))
from llm_palindrome.validator import is_palindrome


def tape(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict[str, object]:
    value = tape(text)
    mismatch = next(
        (
            {"offset": i, "left": value[i], "right": value[-1 - i]}
            for i in range(len(value) // 2)
            if value[i] != value[-1 - i]
        ),
        None,
    )
    left_sha = hashlib.sha256(value.encode()).hexdigest()
    right_sha = hashlib.sha256(value[::-1].encode()).hexdigest()
    return {
        "letters": len(value),
        "two_pointer_exact": bool(value) and mismatch is None,
        "first_mismatch": mismatch,
        "project_validator": bool(is_palindrome(text)),
        "forward_sha256": left_sha,
        "reverse_sha256": right_sha,
        "sha_equal": left_sha == right_sha,
    }


def cumulative_boundaries(words: list[str]) -> dict[int, int]:
    total = 0
    result = {}
    for count, word in enumerate(words, 1):
        total += len(tape(word))
        result[total] = count
    return result


def main() -> dict[str, object]:
    parent_payload = json.loads(PARENT.read_text())
    parent = max(parent_payload["rows"], key=lambda row: row["audit"]["letters"])
    parent_text = parent["rendered"]
    parent_tape = tape(parent_text)
    assert len(parent_tape) == 498
    assert parent_tape == parent_tape[::-1]

    shell_payload = json.loads(SHELL.read_text())
    shell = shell_payload["candidates"][0]
    shell_units = shell["units"]
    left_units = shell_units[: len(shell_units) // 2]
    right_units = shell_units[len(shell_units) // 2 :]
    left_shell = " ".join(left_units)
    right_shell = " ".join(right_units)
    assert tape(left_shell) == tape(right_shell)[::-1]

    words = parent_text.split()
    left_boundaries = cumulative_boundaries(words)
    right_boundaries = cumulative_boundaries(list(reversed(words)))
    joint = sorted(set(left_boundaries) & set(right_boundaries))
    first_joint = joint[0]
    assert first_joint == 129

    frontier = []
    for depth in range(1, first_joint + 1):
        inner = parent_tape[depth : len(parent_tape) - depth]
        child_tape = tape(left_shell) + inner + tape(right_shell)
        frontier.append(
            {
                "depth": depth,
                "left_complete_word_boundary": depth in left_boundaries,
                "right_complete_word_boundary": depth in right_boundaries,
                "both_complete_word_boundaries": depth in left_boundaries and depth in right_boundaries,
                "exact_tape_control": child_tape == child_tape[::-1],
                "control_letters": len(child_tape),
                "inherited_outer_filler_letters_remaining_per_side": first_joint - depth,
            }
        )
    assert all(row["exact_tape_control"] for row in frontier)
    assert not any(row["both_complete_word_boundaries"] for row in frontier[:-1])

    left_count = left_boundaries[first_joint]
    right_count = right_boundaries[first_joint]
    inner_words = words[left_count : len(words) - right_count]
    inner_text = " ".join(inner_words)
    clean_render = " ".join([*left_units, inner_text, *right_units])
    clean_audit = audit(clean_render)
    assert clean_audit["letters"] == 296
    assert clean_audit["two_pointer_exact"]
    assert clean_audit["project_validator"]
    assert clean_audit["sha_equal"]

    selected = [frontier[index - 1] for index in (1, 10, 11, 128, 129)]
    return {
        "experiment_id": "reopen-498-outer-window-20261002",
        "method": "enumerate symmetric normalized cut depths and require complete word boundaries before rendering",
        "parent": {
            "artifact": str(PARENT.relative_to(ROOT)),
            "letters": 498,
            "sha256": hashlib.sha256(parent_tape.encode()).hexdigest(),
            "role": "genuine raw-length incumbent; outer filler is not reader prose",
        },
        "replacement_shell": {
            "artifact": str(SHELL.relative_to(ROOT)),
            "letters": shell["audit"]["letters"],
            "left_units": left_units,
            "right_units": right_units,
            "pair_exact": tape(left_shell) == tape(right_shell)[::-1],
        },
        "stats": {
            "depths_checked": len(frontier),
            "first_joint_complete_word_boundary": first_joint,
            "joint_complete_word_boundaries_through_129": len(
                [row for row in frontier if row["both_complete_word_boundaries"]]
            ),
            "longest_exact_tape_control": max(row["control_letters"] for row in frontier),
            "renderable_exact_rows": 1,
            "admitted_reader_rows": 0,
        },
        "selected_frontier": selected,
        "clean_reopen_control": {
            "depth": first_joint,
            "removed_left_words": words[:left_count],
            "removed_right_words": words[len(words) - right_count :],
            "rendered": clean_render,
            "audit": clean_audit,
            "growth_over_498": clean_audit["letters"] - 498,
            "reader_status": "rejected: exact but only 296 letters and inherits the formulaic 240-letter center",
        },
        "obstruction": {
            "signature": "no bilateral complete-word cut before the full 129-letter filler shell",
            "evidence": "depths 1-128 split at least one inherited word; keeping those spans also retains the word-salad shell",
            "consequence": "incremental complete-clause wrapping cannot repair the 498 incumbent; a readable growth operator must either carry partial-word ownership across the reopened seam or replace the full 258-letter shell with more than 258 exact prose letters",
        },
        "provenance": {
            "finished_tape_reversal": False,
            "posthoc_candidate_repair": False,
            "per_candidate_rlaif": False,
            "reader_certified": False,
        },
        "next_operator": "restart from the 240-letter center with typed prose spans in the live overhang state, or author a greater-than-258-letter exact discourse shell; do not append fixed clauses behind an unchanged residual",
    }


if __name__ == "__main__":
    payload = main()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
