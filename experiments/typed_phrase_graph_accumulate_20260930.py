"""Accumulate an independently streamed outer edge around the 214-letter graph.

This is a lineage experiment, not a readability claim: it preserves the
longer working tape while making the inherited formulaic seam debt explicit.
The outer edge is admitted only through the same character-level obligation
and is independently audited after composition.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from llm_palindrome.validator import is_palindrome

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs" / "typed-phrase-graph-growth-20260929.json"
OUT = ROOT / "runs" / "typed-phrase-graph-accumulate-20260930.json"
LEFT = "Aron saw evil."
RIGHT = "Live was Nora."


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict[str, object]:
    tape = letters(text)
    mismatches = [
        (i, tape[i], tape[-1 - i])
        for i in range(len(tape) // 2)
        if tape[i] != tape[-1 - i]
    ]
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {
        "letters": len(tape),
        "two_pointer_exact": bool(tape) and not mismatches,
        "first_mismatch": mismatches[0] if mismatches else None,
        "validator_exact": is_palindrome(text),
        "sha256_forward": forward,
        "sha256_reverse": reverse,
        "sha_equal": forward == reverse,
    }


def run() -> dict:
    base = json.loads(BASE.read_text())
    base_text = base["candidate"]["text"]
    rendered = f"{LEFT} {base_text} {RIGHT}"
    base_tape = letters(base_text)
    left_tape = letters(LEFT)
    right_tape = letters(RIGHT)
    assert right_tape == left_tape[::-1]
    result = {
        "experiment": "typed_phrase_graph_accumulate_20260930",
        "method": "online outer-edge accumulation over an exact editable phrase graph",
        "candidate": {
            "text": rendered,
            "audit": audit(rendered),
            "provenance": {
                "base_artifact": str(BASE.relative_to(ROOT)),
                "base_letters": len(base_tape),
                "outer_left": LEFT,
                "outer_right": RIGHT,
                "outer_edge_streamed_online": True,
                "new_letters": len(left_tape) + len(right_tape),
                "complete_sentence_sweep": False,
                "finished_tape_reversal": False,
                "catalogue_text": False,
                "posthoc_character_repair": False,
                "repeated_self_palindromic_unit": False,
                "reader_gate": "closed: inherited formulaic seams; no blinded ratings",
            },
            "novelty_preflight": {
                "status": "structural_growth; not a new readability result",
                "new_outer_composition": True,
                "inherited_graph_lineage": True,
                "semordnilap_debt_inherited": True,
            },
        },
        "controls": [
            {"kind": "base_214", "text": base_text, "audit": audit(base_text)},
            {"kind": "smoother_146", "source": "runs/typed-phrase-graph-scene-repair-20260929.json"},
        ],
        "next_construction": {
            "operator": "replace one inherited formulaic edge with a typed natural event pair while keeping both sides variable",
            "debt": "I saw arbitrary reversed-word edges and compressed copular returns",
            "reader_test": "blinded intact-versus-shuffled ratings only after a non-formulaic exact candidate",
        },
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
