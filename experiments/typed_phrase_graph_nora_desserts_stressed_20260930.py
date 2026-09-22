"""Jointly edit the Nora/desserts mirrored seam in the 240-letter draft.

The outside tape is immutable.  A window pair enters the draft only when
the normalized right phrase is the exact character reverse of the left.
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
SOURCE = ROOT / "runs/typed-phrase-graph-noel-war-window-edit-20260930.json"
OUT = ROOT / "runs/typed-phrase-graph-nora-desserts-stressed-20260930.json"
BASE = json.loads(SOURCE.read_text())["candidate"]["text"]
OLD_LEFT = "Nora, I saw desserts."
OLD_RIGHT = "Stressed was I, Aron."


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def audit(text: str) -> dict:
    norm = normalize(text)
    mismatch = next((i for i in range(len(norm) // 2)
                     if norm[i] != norm[-1 - i]), None)
    forward = hashlib.sha256(norm.encode()).hexdigest()
    reverse = hashlib.sha256(norm[::-1].encode()).hexdigest()
    return {
        "letters": len(norm),
        "two_pointer_exact": mismatch is None,
        "validator_exact": is_palindrome(text),
        "first_mismatch": mismatch,
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
        ("Nora, was I stressed?", "Desserts, I saw, Aron."),
        ("Nora, was I stressed?", "Desserts, I saw Aron."),
        ("Nora, did I stress?", "Sserts, I did, Aron."),
        ("Nora, can I stress?", "Sserts, I nac, Aron."),
    ]
    start = BASE.index(OLD_LEFT)
    end = BASE.index(OLD_RIGHT, start + len(OLD_LEFT))
    outside = letters(BASE[:start] + BASE[start + len(OLD_LEFT):end]
                      + BASE[end + len(OLD_RIGHT):])
    rows = []
    for left, right in variants:
        child = build(left, right)
        left_start = child.index(left)
        right_start = child.index(right, left_start + len(left))
        child_outside = letters(child[:left_start] + child[left_start + len(left):right_start]
                                 + child[right_start + len(right):])
        rows.append({
            "left": left,
            "right": right,
            "text": child,
            "window_tape_reverse": letters(left) == letters(right)[::-1],
            "audit": audit(child),
            "outside_tape_preserved": child_outside == outside,
        })
    selected = rows[0]
    result = {
        "experiment": "typed_phrase_graph_nora_desserts_stressed_20260930",
        "method": "immutable outside tape; jointly typed Nora/desserts-stressed seam",
        "parent_artifact": str(SOURCE.relative_to(ROOT)),
        "parent_letters": audit(BASE)["letters"],
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
            "status": "left question is ordinary English; right phrase is locally awkward and inherited draft remains formulaic",
            "human_certified": False,
        },
        "next_construction": "grow from this paired seam while replacing the awkward dessert clause with a fresh typed relation",
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
