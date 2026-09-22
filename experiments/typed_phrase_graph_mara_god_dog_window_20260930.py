"""Test one typed, jointly variable Mara/Dog seam in the 240-letter tape.

The parent tape is immutable outside the two declared clauses.  Candidate
clauses are authored as grammatical-ish units; exactness is checked only
after rendering, with an independent normalized two-pointer audit and SHA
comparison.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_palindrome.validator import is_palindrome, normalize

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs/typed-phrase-graph-nora-evil-live-window-20260930.json"
OUT = ROOT / "runs/typed-phrase-graph-mara-god-dog-window-20260930.json"
LEFT_OLD = "Mara, I saw God."
RIGHT_OLD = "Dog was I, Aram."


def audit(text: str) -> dict:
    tape = normalize(text)
    mismatch = next((i for i in range(len(tape) // 2)
                     if tape[i] != tape[-i - 1]), None)
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {
        "letters": len(tape),
        "two_pointer_exact": mismatch is None,
        "validator_exact": is_palindrome(text),
        "first_mismatch": mismatch,
        "sha256_forward": forward,
        "sha256_reverse": reverse,
        "sha_equal": forward == reverse,
    }


def main() -> None:
    source = json.loads(SOURCE.read_text())
    parent = source["rendered_candidates"][0]["candidate"]
    assert parent.count(LEFT_OLD) == 1 and parent.count(RIGHT_OLD) == 1
    pairs = [
        # Requested natural-question neighborhood.
        ("Mara, was I God?", "Dog, I saw, Aram."),
        # Same tape, with an explicit question/answer-like predicate split.
        ("Mara, was I God?", "Dog is a war, Am."),
        ("Mara, God was I.", "I saw Dog, Aram."),
        ("Mara, I saw God.", "Dog was I, Aram."),
    ]
    rows = []
    for left, right in pairs:
        rendered = parent.replace(LEFT_OLD, left).replace(RIGHT_OLD, right)
        rows.append({
            "window": {"left": left, "right": right},
            "joint_reverse_obligation": normalize(left) == normalize(right)[::-1],
            "candidate": rendered,
            "audit": audit(rendered),
            "outside_tape_preserved": (
                parent.split(LEFT_OLD, 1)[0] == rendered.split(left, 1)[0]
                and parent.rsplit(RIGHT_OLD, 1)[1] == rendered.rsplit(right, 1)[1]
            ),
        })
    exact = [r for r in rows if r["audit"]["two_pointer_exact"]]
    result = {
        "experiment": "typed_phrase_graph_mara_god_dog_window_20260930",
        "method": "immutable 240-letter parent; jointly typed Mara/Dog seam",
        "parent_artifact": str(SOURCE.relative_to(ROOT)),
        "parent_letters": audit(parent)["letters"],
        "window": {"left_old": LEFT_OLD, "right_old": RIGHT_OLD},
        "rendered_candidates": rows,
        "selected": exact[0] if exact else None,
        "provenance": {
            "window_only": True, "outside_tape_unchanged": True,
            "catalogue_text": False, "posthoc_character_repair": False,
            "rlaif_per_candidate": False,
            "reader_gate": "closed: mechanical rough draft; no blinded ratings",
        },
        "readability": {
            "status": "local clauses are more sentence-like, but inherited draft is formulaic",
            "human_certified": False,
        },
        "next_construction": "carry the paired seam into a new clause while replacing repeated I-saw predicates",
    }
    assert exact and all(r["outside_tape_preserved"] for r in exact)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
