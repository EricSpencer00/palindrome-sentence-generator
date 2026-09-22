"""Jointly search an authored clause pair for the 214-letter graph seam.

The seam is opened before composition: each left and right clause is an
independently authored typed template, and the pair is admitted only when the
live character frontier closes.  This is deliberately a small experiment,
not a word-reversal sweep.  The old ``deliver/diaper`` pair is retained as a
scaffold and compared with a new noun/event bank.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from experiments.typed_phrase_graph_growth_20260929 import letters, exact_two_pointer, validator


@dataclass(frozen=True)
class Clause:
    text: str
    role: str
    source: str

    @property
    def tape(self) -> str:
        return letters(self.text)


# These are authored as ordinary clause options, with grammatical roles
# declared before matching.  The right bank is not made by reversing a left
# candidate; it contains separate copular-report templates.
LEFT = (
    Clause("Nora, I saw deliver.", "vocative-perception", "legacy-seam"),
    Clause("Nora, I saw drawer.", "vocative-perception", "fresh-event-bank"),
    Clause("Nora, I saw reward.", "vocative-perception", "fresh-event-bank"),
    Clause("Nora, I saw parts.", "vocative-perception", "fresh-event-bank"),
)
RIGHT = (
    Clause("Reviled was I, Aron.", "copular-report", "legacy-seam"),
    Clause("Reward was I, Aron.", "copular-report", "fresh-event-bank"),
    Clause("Drawer was I, Aron.", "copular-report", "fresh-event-bank"),
    Clause("Strap was I, Aron.", "copular-report", "fresh-event-bank"),
)


def online_pair(left: Clause, right: Clause) -> dict:
    """Return the first mismatch while consuming both clause tapes online."""
    i, j = 0, len(right.tape) - 1
    matched = []
    while i < len(left.tape) and j >= 0 and left.tape[i] == right.tape[j]:
        matched.append(left.tape[i])
        i += 1
        j -= 1
    return {
        "matched": len(matched),
        "complete": i == len(left.tape) and j < 0,
        "left_remaining": left.tape[i:],
        "right_remaining": right.tape[: j + 1],
        "trace": "".join(matched),
    }


def run() -> dict:
    attempts = []
    accepted = []
    for left in LEFT:
        for right in RIGHT:
            frontier = online_pair(left, right)
            row = {"left": left.text, "right": right.text,
                   "left_role": left.role, "right_role": right.role,
                   "sources": [left.source, right.source], **frontier}
            attempts.append(row)
            if frontier["complete"] and left.tape != right.tape:
                accepted.append((left, right, row))

    # Preserve the full 214-letter lineage, replacing only the chosen seam if
    # a new independently authored pair closes.  No post-hoc character edits.
    base = json.loads((Path(__file__).parents[1] / "runs/typed-phrase-graph-growth-20260929.json").read_text())
    base_text = base["candidate"]["text"]
    chosen = accepted[0] if accepted else None
    candidate = base_text
    if chosen:
        old = ("Nora, I saw deliver.", "Noel, I saw diaper.",
               "Repaid was I, Leon.", "Reviled was I, Aron.")
        candidate = base_text.replace(old[0], chosen[0].text).replace(old[3], chosen[1].text)

    tape = letters(candidate)
    result = {
        "method": "typed natural-edge pair search with online opposing frontier",
        "candidate": {
            "text": candidate, "letters": len(tape), "normalized": tape,
            "exact_two_pointer": exact_two_pointer(candidate),
            "validator": validator(candidate),
            "sha256": hashlib.sha256(tape.encode()).hexdigest(),
            "independent_forward_reverse_sha256": hashlib.sha256(tape[::-1].encode()).hexdigest(),
            "provenance": {"base": "typed-phrase-graph-growth-20260929.json",
                           "operator": "replace one typed seam only after online full-pair closure",
                           "catalogue_text": False, "posthoc_character_repair": False,
                           "accepted_pair": [chosen[0].text, chosen[1].text] if chosen else None},
            "reader_gate": "pending-blinded-human-ratings; legacy formulaic scaffold disclosed",
        },
        "attempts": attempts,
        "accepted_pairs": [r for _, _, r in accepted],
        "residual_frontier": "No fresh pair closed; the authored bank reaches only partial tape matches and the legacy deliver/reviled seam remains.",
        "next_operator": "author a transitive active/passive event pair with a shared noun across the seam, then intersect its clause prefixes before adding it to the scaffold",
    }
    out = Path(__file__).parents[1] / "runs/natural-edge-pair-20260930.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    r = run()
    print(r["candidate"]["text"])
    print(json.dumps({k: r["candidate"][k] for k in ("letters", "exact_two_pointer", "validator", "sha256")}, indent=2))
