"""Bounded paired edit of the Sara/Aras seam in the 238-letter incumbent.

The complete tape is inherited unchanged except for one mirrored window.  The
search enumerates a small typed question/answer family and admits only pairs
whose normalized character tapes are exact reverses.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_palindrome.validator import is_palindrome, normalize


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/typed-phrase-graph-sara-window-edit-20260930.json"

BASE = json.loads(
    (ROOT / "runs/typed-phrase-graph-outer-question-answer-20260930.json").read_text()
)["candidate"]["text"]
OLD_LEFT = "Sara, I saw live."
OLD_RIGHT = "Evil was I, Aras."
OLD_WINDOW = OLD_LEFT + " " + OLD_RIGHT


def tape(text: str) -> str:
    return re.sub(r"[^a-zA-Z]", "", text).lower()


def audit(text: str) -> dict:
    norm = normalize(text)
    pointer = all(norm[i] == norm[-1 - i] for i in range(len(norm) // 2))
    forward = hashlib.sha256(norm.encode()).hexdigest()
    reverse = hashlib.sha256(norm[::-1].encode()).hexdigest()
    return {
        "letters": len(norm),
        "two_pointer_exact": pointer,
        "validator_exact": is_palindrome(text),
        "first_mismatch": next(
            (i for i in range(len(norm) // 2) if norm[i] != norm[-1 - i]), None
        ),
        "sha256_forward": forward,
        "sha256_reverse": reverse,
        "sha_equal": forward == reverse,
    }


def build(left: str, right: str) -> str:
    left_start = BASE.index(OLD_LEFT)
    right_start = BASE.index(OLD_RIGHT, left_start + len(OLD_LEFT))
    return BASE[:left_start] + left + BASE[left_start + len(OLD_LEFT) : right_start] + right + BASE[right_start + len(OLD_RIGHT) :]


def main() -> None:
    # The first is the requested ordinary-English question/answer edit.  The
    # others are nearby typed variants, included to document the live seam
    # domain rather than silently selecting a post-hoc repair.
    variants = [
        ("Sara, did I live?", "Evil, I did, Aras."),
        ("Sara, can I live?", "Evil, I nac, Aras."),
        ("Sara, may I live?", "Evil, I yam, Aras."),
        ("Aras, did I live?", "Evil, I did, Sara."),
        ("Sara, did I live!", "Evil, I did, Aras."),
    ]
    rows = []
    for left, right in variants:
        window_ok = tape(left) == tape(right)[::-1]
        text = build(left, right)
        left_start = BASE.index(OLD_LEFT)
        right_start = BASE.index(OLD_RIGHT, left_start + len(OLD_LEFT))
        outside_parent = tape(BASE[:left_start] + BASE[left_start + len(OLD_LEFT) : right_start] + BASE[right_start + len(OLD_RIGHT) :])
        left_new = text.index(left)
        right_new = text.index(right, left_new + len(left))
        outside_child = tape(text[:left_new] + text[left_new + len(left) : right_new] + text[right_new + len(right) :])
        rows.append(
            {
                "left": left,
                "right": right,
                "text": text,
                "window_tape_reverse": window_ok,
                "audit": audit(text),
                "outside_tape_preserved": outside_child == outside_parent,
            }
        )
    selected = next(row for row in rows if row["left"] == "Sara, did I live?")
    result = {
        "experiment": "typed_phrase_graph_sara_window_edit_20260930",
        "method": "immutable outside tape; jointly typed Sara/Aras question-answer window",
        "parent_artifact": "runs/typed-phrase-graph-outer-question-answer-20260930.json",
        "parent_letters": 238,
        "window": {"old": [OLD_LEFT, OLD_RIGHT], "new": [selected["left"], selected["right"]]},
        "candidate": selected,
        "variants": rows,
        "provenance": {
            "outside_tape_preserved": selected["outside_tape_preserved"],
            "joint_character_obligation": True,
            "posthoc_character_repair": False,
            "catalogue_text": False,
            "reader_gate": "closed: exact rough draft; no blinded ratings",
        },
        "readability": {
            "status": "local question/answer is ordinary English; inherited draft remains formulaic",
            "human_certified": False,
        },
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
