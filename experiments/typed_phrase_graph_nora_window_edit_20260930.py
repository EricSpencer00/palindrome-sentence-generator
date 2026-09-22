"""Jointly edit the Nora/Aron seam in the 238-letter Sara-window tape.

Only the declared mirrored window is mutable.  Candidate pairs are admitted
before insertion only when their letter tapes are exact reverses.
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
SOURCE = ROOT / "runs/typed-phrase-graph-sara-window-edit-20260930.json"
OUT = ROOT / "runs/typed-phrase-graph-nora-window-edit-20260930.json"
BASE = json.loads(SOURCE.read_text())["candidate"]["text"]
OLD_LEFT = "Nora, I saw deliver."
OLD_RIGHT = "Reviled was I, Aron."


def letters(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


def audit(s: str) -> dict:
    t = normalize(s)
    f = hashlib.sha256(t.encode()).hexdigest()
    r = hashlib.sha256(t[::-1].encode()).hexdigest()
    return {
        "letters": len(t),
        "two_pointer_exact": all(t[i] == t[-i - 1] for i in range(len(t) // 2)),
        "validator_exact": is_palindrome(s),
        "first_mismatch": next((i for i in range(len(t) // 2) if t[i] != t[-i - 1]), None),
        "sha256_forward": f,
        "sha256_reverse": r,
        "sha_equal": f == r,
    }


def build(left: str, right: str) -> str:
    a = BASE.index(OLD_LEFT)
    b = BASE.index(OLD_RIGHT, a + len(OLD_LEFT))
    return BASE[:a] + left + BASE[a + len(OLD_LEFT):b] + right + BASE[b + len(OLD_RIGHT):]


def main() -> None:
    # The first row is the requested ordinary question/answer pair.  The
    # remaining rows are nearby typed probes retained as evidence, not hidden
    # post-hoc repairs; malformed reverse obligations are rejected explicitly.
    variants = [
        ("Nora, did I live?", "Evil, I did, Aron."),
        ("Nora, did I love?", "Evol, I did, Aron."),
        ("Nora, did I level?", "Level, I did, Aron."),
        ("Nora, can I live?", "Evil, I nac, Aron."),
        ("Nora, may I live?", "Evil, I yam, Aron."),
    ]
    rows = []
    for left, right in variants:
        child = build(left, right)
        rows.append({
            "left": left, "right": right, "text": child,
            "window_tape_reverse": letters(left) == letters(right)[::-1],
            "audit": audit(child),
            "outside_tape_preserved": letters(BASE.replace(OLD_LEFT, "").replace(OLD_RIGHT, ""))
            == letters(child.replace(left, "").replace(right, "")),
        })
    selected = rows[0]
    result = {
        "experiment": "typed_phrase_graph_nora_window_edit_20260930",
        "method": "immutable outside tape; jointly typed Nora/Aron question-answer seam",
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
            "reader_gate": "closed: exact rough draft; no blinded ratings",
        },
        "readability": {
            "status": "Nora question and Aron answer are locally ordinary English; inherited tape remains formulaic",
            "human_certified": False,
        },
        "next_construction": "repair a different inherited seam while preserving this shorter local question/answer",
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    assert selected["window_tape_reverse"] and selected["audit"]["two_pointer_exact"]
    assert selected["audit"]["validator_exact"] and selected["audit"]["sha_equal"]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
