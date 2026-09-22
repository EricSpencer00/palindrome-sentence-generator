"""Jointly edit the Nora/Live mirrored seam in the 240-letter tape.

The parent is immutable except for the declared two-clause window.  Each
candidate is authored as a typed left/right pair and is admitted only when
the rendered character tapes are exact reverses; no character-level repair is
performed after rendering.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_palindrome.validator import is_palindrome, normalize

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs/typed-phrase-graph-noel-stressed-desserts-edit-20260930.json"
OUT = ROOT / "runs/typed-phrase-graph-nora-evil-live-window-20260930.json"

LEFT_OLD = "Nora, I saw evil."
RIGHT_OLD = "Live was I, Aron."


def audit(text: str) -> dict:
    tape = normalize(text)
    rev = tape[::-1]
    mismatch = next((i for i in range(len(tape) // 2) if tape[i] != tape[-i - 1]), None)
    return {
        "letters": len(tape),
        "two_pointer_exact": mismatch is None,
        "validator_exact": is_palindrome(text),
        "first_mismatch": mismatch,
        "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(rev.encode()).hexdigest(),
        "sha_equal": hashlib.sha256(tape.encode()).hexdigest()
        == hashlib.sha256(rev.encode()).hexdigest(),
    }


def main() -> None:
    source = json.loads(SOURCE.read_text())
    parent = source["candidate"]["text"]
    assert parent.count(LEFT_OLD) == 1 and parent.count(RIGHT_OLD) == 1

    # The first pair is the requested joint question/answer-style edit.  The
    # other pairs are a deliberately small typed neighborhood, not a bank of
    # reversed words: all are complete local clauses and are tested together.
    pairs = [
        ("Nora, was I evil?", "Live, I saw, Aron."),
        ("Nora, did I see evil?", "Live, I did see, Aron."),
        ("Nora, I saw evil.", "Live was I, Aron."),
        ("Nora, saw I evil.", "Live was I, Aron."),
    ]
    rows = []
    for left, right in pairs:
        rendered = parent.replace(LEFT_OLD, left).replace(RIGHT_OLD, right)
        rows.append(
            {
                "window": {"left": left, "right": right},
                "joint_reverse_obligation": normalize(left) == normalize(right)[::-1],
                "candidate": rendered,
                "audit": audit(rendered),
                "outside_tape_preserved": (
                    parent.split(LEFT_OLD, 1)[0] == rendered.split(left, 1)[0]
                    and parent.rsplit(RIGHT_OLD, 1)[1] == rendered.rsplit(right, 1)[1]
                ),
            }
        )

    exact = [r for r in rows if r["audit"]["two_pointer_exact"]]
    result = {
        "experiment": "typed_phrase_graph_nora_evil_live_window_20260930",
        "method": "immutable outside tape; jointly typed Nora/Live seam variants",
        "parent_artifact": str(SOURCE.relative_to(ROOT)),
        "parent_letters": audit(parent)["letters"],
        "rendered_candidates": rows,
        "selected": exact[0] if exact else None,
        "provenance": {
            "window_only": True,
            "catalogue_text": False,
            "posthoc_character_repair": False,
            "rlaif_per_candidate": False,
            "reader_gate": "closed: mechanical rough draft; no blinded ratings",
        },
        "readability": {
            "status": "local question punctuation is more natural, but inherited draft remains formulaic",
            "human_certified": False,
        },
        "next_construction": "retain exact seam and grow from an adjacent paired clause while diversifying repeated I-saw predicates",
    }
    assert exact, "requested seam neighborhood must include an exact jointly typed pair"
    assert exact[0]["outside_tape_preserved"]
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
