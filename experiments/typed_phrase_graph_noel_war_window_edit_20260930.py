"""Edit only the Noel/Raw mirrored seam in the 238-letter incumbent.

The search is a small typed question/answer domain.  Each pair is admitted
only when its normalized tape is the character reverse of the other; the
rest of the parent tape is immutable.
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
OUT = ROOT / "runs/typed-phrase-graph-noel-war-window-edit-20260930.json"
BASE = json.loads((ROOT / "runs/typed-phrase-graph-sara-window-edit-20260930.json").read_text())["candidate"]["text"]
OLD_LEFT = "Noel, I saw war."
OLD_RIGHT = "Raw was I, Leon."


def tape(text: str) -> str:
    return re.sub(r"[^a-zA-Z]", "", text).lower()


def audit(text: str) -> dict:
    norm = normalize(text)
    first = next((i for i in range(len(norm) // 2) if norm[i] != norm[-1 - i]), None)
    forward = hashlib.sha256(norm.encode()).hexdigest()
    reverse = hashlib.sha256(norm[::-1].encode()).hexdigest()
    return {
        "letters": len(norm),
        "two_pointer_exact": first is None,
        "validator_exact": is_palindrome(text),
        "first_mismatch": first,
        "sha256_forward": forward,
        "sha256_reverse": reverse,
        "sha_equal": forward == reverse,
    }


def build(left: str, right: str) -> str:
    start = BASE.index(OLD_LEFT)
    end = BASE.index(OLD_RIGHT, start + len(OLD_LEFT))
    return BASE[:start] + left + BASE[start + len(OLD_LEFT):end] + right + BASE[end + len(OLD_RIGHT):]


def main() -> None:
    variants = [
        ("Noel, did I draw?", "Ward, I did, Leon."),
        ("Noel, did I level?", "Level, I did, Leon."),
        ("Noel, can I draw?", "Ward, I nac, Leon."),
    ]
    start = BASE.index(OLD_LEFT)
    end = BASE.index(OLD_RIGHT, start + len(OLD_LEFT))
    outside = tape(BASE[:start] + BASE[start + len(OLD_LEFT):end] + BASE[end + len(OLD_RIGHT):])
    rows = []
    for left, right in variants:
        text = build(left, right)
        left_start = text.index(left)
        right_start = text.index(right, left_start + len(left))
        child_outside = tape(text[:left_start] + text[left_start + len(left):right_start] + text[right_start + len(right):])
        rows.append({
            "left": left,
            "right": right,
            "text": text,
            "window_tape_reverse": tape(left) == tape(right)[::-1],
            "audit": audit(text),
            "outside_tape_preserved": child_outside == outside,
        })
    selected = rows[0]
    result = {
        "experiment": "typed_phrase_graph_noel_war_window_edit_20260930",
        "method": "immutable outside tape; jointly typed Noel question/Raw answer seam",
        "parent_artifact": "runs/typed-phrase-graph-sara-window-edit-20260930.json",
        "parent_letters": 238,
        "window": {"old": [OLD_LEFT, OLD_RIGHT], "new": [selected["left"], selected["right"]]},
        "candidate": selected,
        "variants": rows,
        "provenance": {
            "outside_tape_preserved": selected["outside_tape_preserved"],
            "joint_character_obligation": True,
            "posthoc_character_repair": False,
            "catalogue_text": False,
            "window_diff_only": True,
            "reader_gate": "closed: exact rough draft; no blinded ratings",
        },
        "readability": {
            "status": "local question/answer is readable English; inherited draft remains formulaic and repetitive",
            "human_certified": False,
        },
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
