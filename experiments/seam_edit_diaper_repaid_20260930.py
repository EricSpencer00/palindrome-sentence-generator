"""Edit one mirrored window of the 236-letter accumulator.

The outside tape is byte-for-byte inherited.  Only the diaper/repaid window is
replaced; exactness is checked by two independent implementations.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

BASE = json.loads(Path("runs/typed-phrase-graph-accumulate-20260930.json").read_text())["candidate"]["text"]
OLD = "Noel, I saw diaper. Repaid was I, Leon."
NEW = "Noel, I saw live. Evil was I, Leon."
assert BASE.count(OLD) == 1


def letters(s: str) -> str:
    return "".join(re.findall("[a-z]", s.lower()))


def audit(s: str) -> dict:
    tape = letters(s)
    mismatches = [(i, tape[i], tape[-1 - i]) for i in range(len(tape) // 2) if tape[i] != tape[-1 - i]]
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {
        "letters": len(tape),
        "two_pointer_exact": not mismatches,
        "first_mismatch": mismatches[0] if mismatches else None,
        "sha256_forward": forward,
        "sha256_reverse": reverse,
        "sha_equal": forward == reverse,
    }


candidate = BASE.replace(OLD, NEW)
left, right = BASE.split(OLD)
row = {
    "text": candidate,
    "audit": audit(candidate),
    "provenance": {
        "parent": "runs/typed-phrase-graph-accumulate-20260930.json",
        "parent_letters": audit(BASE)["letters"],
        "edited_window_before": OLD,
        "edited_window_after": NEW,
        "outside_tape_unchanged": left + right == candidate.replace(NEW, "", 1),
        "method": "joint mirrored-window substitution; no post-hoc character repair",
        "reader_status": "unrated rough draft; left clause is more ordinary, but repeats live/evil lineage",
    },
}
OUT = Path("runs/seam-edit-diaper-repaid-20260930.json")
OUT.write_text(json.dumps({"experiment": "seam_edit_diaper_repaid_20260930", "candidate": row}, indent=2) + "\n")
print(json.dumps(row, indent=2))
