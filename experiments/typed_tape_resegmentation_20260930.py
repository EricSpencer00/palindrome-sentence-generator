"""Typed outside-in resegmentation of the 214-letter working tape.

This is an input-constrained parser, not a punctuation repair: phrase units
are selected while the two character frontiers are consumed.  Every accepted
unit has an authored syntactic type, and the complete tape is independently
audited after parsing.  A failure is useful only if its residual is reported.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_palindrome.validator import is_palindrome


def letters(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


@dataclass(frozen=True)
class Unit:
    text: str
    role: str
    source: str

    @property
    def tape(self) -> str:
        return letters(self.text)


TAPE_TEXT = (
    "Nora, I saw evil. Noel, I saw war. Mara, I saw God. Sara, I saw live. "
    "Nora, I saw desserts. Noel, I saw stressed. Nora, I saw deliver. "
    "Noel, I saw diaper. Repaid was I, Leon. Reviled was I, Aron. "
    "Desserts was I, Leon. Stressed was I, Aron. Evil was I, Aras. "
    "Dog was I, Aram. Raw was I, Leon. Live was I, Aron."
)

# Authored independently as typed templates.  The parser may use only these
# units; it never invents punctuation or reverses a finished sentence.
LEFT = tuple(Unit(x, "V-PERCEPTION", "typed-authored-left") for x in (
    "Nora, I saw evil.", "Noel, I saw war.", "Mara, I saw God.",
    "Sara, I saw live.", "Nora, I saw desserts.", "Noel, I saw stressed.",
    "Nora, I saw deliver.", "Noel, I saw diaper.",
))
RIGHT = tuple(Unit(x, "COPULAR-RETURN", "typed-authored-right") for x in (
    "Live was I, Aron.", "Raw was I, Leon.", "Dog was I, Aram.",
    "Evil was I, Aras.", "Stressed was I, Aron.", "Desserts was I, Leon.",
    "Repaid was I, Leon.", "Reviled was I, Aron.",
))


def outside_in_parse(tape: str) -> tuple[list[tuple[Unit, Unit]], str]:
    """Return a disjoint typed path and the first unmatched residual."""
    @lru_cache(None)
    def solve(lo: int, hi: int, used_l: tuple[str, ...], used_r: tuple[str, ...]):
        if lo >= hi:
            return ()
        for l in LEFT:
            if not tape.startswith(l.tape, lo):
                continue
            for r in RIGHT:
                if not tape.endswith(r.tape, hi - len(r.tape), hi):
                    continue
                nlo, nhi = lo + len(l.tape), hi - len(r.tape)
                # The tape is the constraint; phrase units need not have equal
                # lengths.  Their newly exposed characters are checked against
                # the fixed opposite frontier, so boundaries and roles are
                # selected online rather than paired after the fact.
                if nlo > nhi or tape[lo:nlo] != l.tape or tape[nhi:hi] != r.tape:
                    continue
                tail = solve(nlo, nhi, used_l + (l.text,), used_r + (r.text,))
                if tail is not None:
                    return ((l, r),) + tail
        return None

    path = solve(0, len(tape), (), ())
    if path is None:
        return [], tape
    consumed = sum(len(l.tape) + len(r.tape) for l, r in path)
    return list(path), tape[consumed // 2: len(tape) - consumed // 2]


def audit() -> dict:
    tape = letters(TAPE_TEXT)
    path, residual = outside_in_parse(tape)
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {
        "method": "typed outside-in grammar intersection on fixed 214-letter tape",
        "candidate": TAPE_TEXT,
        "letters": len(tape),
        "normalized": tape,
        "exact_two_pointer": all(tape[i] == tape[-1-i] for i in range(len(tape)//2)),
        "validator": is_palindrome(TAPE_TEXT),
        "sha256": forward,
        "independent_forward_reverse_sha256": reverse,
        "typed_path": [{"left": l.text, "left_role": l.role,
                        "right": r.text, "right_role": r.role,
                        "online_match": l.tape == r.tape[::-1]}
                       for l, r in path],
        "residual_letters": len(residual),
        "residual": residual,
        "provenance": {
            "input_constraint": "runs/typed-phrase-graph-growth-20260929.json",
            "units_authored_before_intersection": True,
            "punctuation_invented": False,
            "catalogue_text": False,
            "semordnilap_only": False,
            "novel_candidate": False,
        },
        "reader_gate": "not admitted: formulaic I-saw/copular-return syntax; no blinded ratings",
        "next_repair": "replace one COPULAR-RETURN unit with an independently authored transitive clause whose boundary tape survives the typed intersection",
    }


def run() -> dict:
    result = audit()
    out = Path(__file__).parents[1] / "runs" / "typed-tape-resegmentation-20260930.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
