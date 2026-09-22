"""Bounded paired seam edit on the 240-letter typed phrase-graph draft."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "runs/typed-phrase-graph-noel-war-window-edit-20260930.json"
OUT = ROOT / "runs/typed-phrase-graph-noel-stressed-desserts-edit-20260930.json"
OLD_LEFT = "Noel, I saw stressed."
OLD_RIGHT = "Desserts was I, Leon."
NEW_LEFT = "Noel, was I stressed?"
NEW_RIGHT = "Desserts, I saw, Leon."


def tape(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def audit(text: str) -> dict:
    letters = tape(text)
    forward = hashlib.sha256(letters.encode()).hexdigest()
    reverse = hashlib.sha256(letters[::-1].encode()).hexdigest()
    mismatch = next((i for i, (a, b) in enumerate(zip(letters, letters[::-1])) if a != b), None)
    return {
        "letters": len(letters),
        "two_pointer_exact": all(letters[i] == letters[-1 - i] for i in range(len(letters) // 2)),
        "first_mismatch": mismatch,
        "sha256_forward": forward,
        "sha256_reverse": reverse,
        "sha_equal": forward == reverse,
    }


def main() -> None:
    parent = json.loads(PARENT.read_text())
    text = parent["candidate"]["text"]
    assert text.count(OLD_LEFT) == 1 and text.count(OLD_RIGHT) == 1
    child = text.replace(OLD_LEFT, NEW_LEFT).replace(OLD_RIGHT, NEW_RIGHT)
    result = {
        "experiment": "typed_phrase_graph_noel_stressed_desserts_edit_20260930",
        "method": "immutable outside tape; jointly typed Noel/stressed-desserts seam",
        "parent_artifact": str(PARENT.relative_to(ROOT)),
        "parent_letters": len(tape(text)),
        "window_diff": {"old": [OLD_LEFT, OLD_RIGHT], "new": [NEW_LEFT, NEW_RIGHT]},
        "candidate": {"text": child, "audit": audit(child)},
        "outside_tape_preserved": tape(text[: text.index(OLD_LEFT)]) + tape(text[text.index(OLD_LEFT)+len(OLD_LEFT):text.index(OLD_RIGHT)]) + tape(text[text.index(OLD_RIGHT)+len(OLD_RIGHT):]) == tape(child[: child.index(NEW_LEFT)]) + tape(child[child.index(NEW_LEFT)+len(NEW_LEFT):child.index(NEW_RIGHT)]) + tape(child[child.index(NEW_RIGHT)+len(NEW_RIGHT):]),
        "provenance": {"window_diff_only": True, "catalogue_text": False, "posthoc_character_repair": False, "reader_gate": "closed: exact rough draft; no blinded ratings"},
        "readability": {"status": "question form is more natural locally; surrounding draft remains formulaic and contains inherited syntax debt", "human_certified": False},
        "next_construction": "grow from this preserved seam while replacing one inherited I-saw/reversed-predicate pair with a fresh typed relation",
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["candidate"]["audit"], indent=2))
    print(child)


if __name__ == "__main__":
    main()
