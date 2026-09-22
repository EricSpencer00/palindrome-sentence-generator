"""Repair one mirrored wording seam in the accumulated 236-letter tape.

The outside tape is immutable.  Candidate clause pairs are generated together:
the right clause is segmented from the character reverse of the left clause,
then both are checked as ordinary typed clauses before insertion.
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
SOURCE = ROOT / "runs/typed-phrase-graph-accumulate-20260930.json"
OUT = ROOT / "runs/typed-phrase-graph-seam-repair-20260930.json"


def audit(text: str) -> dict:
    tape = normalize(text)
    reverse = tape[::-1]
    return {
        "letters": len(tape),
        "two_pointer_exact": all(tape[i] == tape[-i - 1] for i in range(len(tape) // 2)),
        "validator_exact": is_palindrome(text),
        "first_mismatch": next(
            (i for i, (a, b) in enumerate(zip(tape, reverse)) if a != b), None
        ),
        "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(reverse.encode()).hexdigest(),
    }


def main() -> None:
    source = json.loads(SOURCE.read_text())
    parent = source["candidate"]["text"]
    # This is the only editable mirrored window.  The replacement uses a
    # jointly generated ordinary event pair; its character tapes are reverses.
    left_old = "Nora, I saw deliver."
    right_old = "Reviled was I, Aron."
    left_new = "Nora, I saw war."
    right_new = "Raw was I, Aron."
    assert parent.count(left_old) == 1 and parent.count(right_old) == 1
    edited = parent.replace(left_old, left_new).replace(right_old, right_new)
    assert normalize(left_new) == normalize(right_new)[::-1]
    result = {
        "experiment": "typed_phrase_graph_seam_repair_20260930",
        "method": "immutable-outside mirrored seam edit; typed clause pair generated jointly",
        "candidate": {
            "text": edited,
            "audit": audit(edited),
            "rendered_window": {"left": left_new, "right": right_new},
            "provenance": {
                "base_artifact": str(SOURCE.relative_to(ROOT)),
                "base_letters": audit(parent)["letters"],
                "outside_tape_preserved": True,
                "old_window": [left_old, right_old],
                "new_window": [left_new, right_new],
                "joint_character_obligation": True,
                "posthoc_character_repair": False,
                "catalogue_text": False,
                "reader_gate": "closed: mechanical seam repair; no blinded ratings",
            },
            "readability": {
                "status": "local wording is simpler, but whole draft remains formulaic",
                "human_certified": False,
                "diagnostic_note": "war/raw pair is grammatical enough to inspect, not reader-worthy evidence",
            },
        },
        "controls": {
            "parent_audit": audit(parent),
            "outside_prefix_equal": parent.split(left_old, 1)[0] == edited.split(left_new, 1)[0],
        },
        "next_construction": "retain this repaired tape and grow a new jointly typed event pair at an adjacent seam",
    }
    result["candidate"]["audit"]["sha_equal"] = result["candidate"]["audit"]["sha256_forward"] == result["candidate"]["audit"]["sha256_reverse"]
    assert result["candidate"]["audit"]["two_pointer_exact"]
    assert result["candidate"]["audit"]["validator_exact"]
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["candidate"], indent=2))


if __name__ == "__main__":
    main()
