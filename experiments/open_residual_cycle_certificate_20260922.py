"""Reproducible certificate for scalable open-residual ABBA growth.

The fixture is deliberately synthetic.  It proves the graph operator can pump
an exact dual parse without passing through an intermediate empty residual; it
does not claim English readability or a new reader candidate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.recursive_product import Edge, materialize_pump, search, tape


def pointer_audit(text: str) -> dict:
    letters = tape(text)
    exact = all(letters[i] == letters[-1 - i]
                for i in range(len(letters) // 2))
    return {
        "letters": len(letters),
        "exact": exact,
        "sha256_forward": hashlib.sha256(letters.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(letters[::-1].encode()).hexdigest(),
    }


def run() -> dict:
    left = {
        "S": (Edge("P", "ab", "A"),),
        "P": (Edge("P", "cb", "B"), Edge("F", "x", "A-prime")),
    }
    # Right transitions are outside-in.  materialize_pump reverses their word
    # order for normal rendering.
    right = {
        "S": (Edge("Q", "a", "A"),),
        "Q": (Edge("Q", "cb", "B-prime"), Edge("F", "xb", "A-prime")),
    }
    report = search(left, right, reject_intermediate_closure=True)
    if not report.pumpable_cycles:
        raise AssertionError("expected a coaccessible nonempty-residual cycle")
    pump = report.pumpable_cycles[0]
    rows = []
    for repetitions in range(5):
        witness = materialize_pump(pump, repetitions)
        rendered = " ".join(witness.left_words + witness.right_words)
        audit = pointer_audit(rendered)
        if not audit["exact"] or audit["sha256_forward"] != audit["sha256_reverse"]:
            raise AssertionError("pump reconstruction lost exactness")
        rows.append({
            "repetitions": repetitions,
            "rendered": rendered,
            "left_words": list(witness.left_words),
            "right_words": list(witness.right_words),
            "left_boundaries": list(witness.left_boundaries),
            "reflected_right_boundaries": list(witness.reflected_right_boundaries),
            "audit": audit,
        })
    return {
        "experiment_id": "open-residual-cycle-certificate-20260922",
        "method": "reachable/coaccessible recursive dual-grammar SCC at identical nonempty residual",
        "synthetic_operator_certificate": True,
        "english_candidate": False,
        "reader_candidates": [],
        "cycle_states": [
            {"left": state.left, "right": state.right,
             "owner": state.owner, "residual": state.residual}
            for state in pump.states
        ],
        "intermediate_empty_closures": report.intermediate_empty_closures,
        "rows": rows,
        "provenance": {
            "catalogue_text": False,
            "finished_tape_reversal": False,
            "post_hoc_repair": False,
            "cycle_residual_never_empty": all(state.residual for state in pump.states),
            "right_parse_reconstructed_independently": True,
        },
        "status": "operator_verified_no_english_result",
        "next_experiment": "Compile authored A-to-B and B-prime-to-A-prime paragraph grammars into this product and search for a coaccessible nonempty-debt SCC; expand only the first unsupported grammar frontier.",
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
