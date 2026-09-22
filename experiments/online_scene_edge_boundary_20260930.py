"""Online typed scene-edge boundary test.

This is a deliberately small, authored edge grammar rather than a Cartesian
sentence sweep.  At each step the left clause chooses a semantic slot and the
right clause must consume its live reverse-character obligation.  The
candidate is rendered only after the two typed edges close.  The purpose of
this lane is to test whether replacing the copular/semordnilap seam with a
normal transitive scene edge can close without post-hoc tape editing.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from llm_palindrome.validator import is_palindrome as independent_validator


def normalize(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def two_pointer(tape: str) -> bool:
    i, j = 0, len(tape) - 1
    while i < j:
        if tape[i] != tape[j]:
            return False
        i += 1
        j -= 1
    return bool(tape)


@dataclass(frozen=True)
class Edge:
    text: str
    subject: str
    verb: str
    object: str
    side: str

    @property
    def tape(self) -> str:
        return normalize(self.text)


# These are independently authored ordinary scene clauses.  They are a
# typed seam bank, not reversed strings and not imported catalogue sentences.
LEFT = (
    Edge("Diana reads a map.", "Diana", "reads", "a map", "left"),
    Edge("Mara opens the door.", "Mara", "opens", "the door", "left"),
    Edge("Nora carries a red book.", "Nora", "carries", "a red book", "left"),
    Edge("Aron hears the bell.", "Aron", "hears", "the bell", "left"),
    Edge("Sara paints a small boat.", "Sara", "paints", "a small boat", "left"),
)
RIGHT = (
    Edge("Mara reads the map.", "Mara", "reads", "the map", "right"),
    Edge("Aron opens the door.", "Aron", "opens", "the door", "right"),
    Edge("Nora carries a blue book.", "Nora", "carries", "a blue book", "right"),
    Edge("Diana hears the bell.", "Diana", "hears", "the bell", "right"),
    Edge("Sara paints a small boat.", "Sara", "paints", "a small boat", "right"),
)


def live_intersection(left: Edge, right_options: tuple[Edge, ...]) -> dict:
    """Consume outside-in characters while selecting one typed right edge."""
    chosen = []
    for right in right_options:
        i, j = 0, len(right.tape) - 1
        while i < len(left.tape) and j >= 0 and left.tape[i] == right.tape[j]:
            i += 1
            j -= 1
        chosen.append({
            "right": right.text,
            "matched": i,
            "left_length": len(left.tape),
            "right_length": len(right.tape),
            "left_residual": left.tape[i:],
            "right_residual": right.tape[: j + 1],
            "closed": i == len(left.tape) == len(right.tape),
        })
    return {"left": left.text, "options": chosen}


def run() -> dict:
    attempts = [live_intersection(left, RIGHT) for left in LEFT]
    closed = [
        (left, right)
        for left in LEFT
        for right in RIGHT
        if left.tape == right.tape[::-1]
    ]
    candidates = []
    for left, right in closed:
        text = f"{left.text} {right.text}"
        tape = normalize(text)
        candidates.append({
            "text": text,
            "letters": len(tape),
            "exact_two_pointer": two_pointer(tape),
            "validator": independent_validator(text),
            "sha256": hashlib.sha256(tape.encode()).hexdigest(),
            "reverse_sha256": hashlib.sha256(tape[::-1].encode()).hexdigest(),
            "provenance": {"typed_online_selection": True, "catalogue_text": False,
                            "posthoc_repair": False, "repeated_unit": False},
            "reader_gate": "pending-blinded-human-ratings",
        })
    result = {
        "method": "online typed transitive scene-edge boundary intersection",
        "candidate_count": len(candidates),
        "candidates": candidates,
        "attempts": attempts,
        "novelty_preflight": {"new_edge_bank": True, "not_complete_sentence_cartesian_sweep": True,
                              "not_semordnilap_only": True},
        "status": "no exact pair in the fresh ordinary scene-edge bank",
        "next_repair": "retain the Diana argument as a live slot and author a finite verb/object pair whose reverse boundary begins with an ordinary English determiner, not a reversed verb",
    }
    out = Path(__file__).parents[1] / "runs" / "online-scene-edge-boundary-20260930.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
