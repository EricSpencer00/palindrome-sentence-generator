"""Rejected complete-clause-atomic probe for an open-residual paragraph cycle.

This experiment deliberately gives the product *complete clauses* as edges.  It
does not import the site's catalogue or freeze a target tape.  A cycle is useful
only when it preserves non-empty character debt; an empty debt before both
grammars finish is rejected by ``recursive_product.search``.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_palindrome.recursive_product import Edge, Report, search, materialize_pump

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/natural-dialogue-open-cycle-20260922.json"

# Independent, authored dialogue clauses.  The old naturalist seam suggested
# looking for acknowledgement/observation turns; these are new clauses, not
# borrowed candidate strings.
LEFT = {
    "S": (Edge("Q", "The lantern marks the harbor.", "observation", "left-A"),),
    "Q": (Edge("Q", "A patient sailor listens.", "acknowledgement", "left-B"),
          Edge("F", "The tide turns at dawn.", "return", "left-C")),
}
RIGHT = {
    "S": (Edge("Q", "At dawn the tide turns.", "return", "right-C"),),
    "Q": (Edge("Q", "The sailor listens patiently.", "acknowledgement", "right-B"),
          Edge("F", "The harbor marks the lantern.", "observation", "right-A")),
}


def letters(text: str) -> str:
    return "".join(re.findall(r"[a-z]", text.casefold()))


def audit(text: str) -> dict:
    tape = letters(text)
    mismatches = [(i, tape[i], tape[-1 - i])
                  for i in range(len(tape) // 2) if tape[i] != tape[-1 - i]]
    return {
        "letters": len(tape),
        "two_pointer_exact": bool(tape) and not mismatches,
        "first_mismatches": mismatches[:4],
        "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
        "sha256_reverse_obligation": hashlib.sha256(tape[::-1].encode()).hexdigest(),
    }


def render(witness) -> str:
    return " ".join((*witness.left_words, *witness.right_words))


def anti_shortcut(text: str, witness) -> dict:
    words = [w.casefold() for w in re.findall(r"[A-Za-z]+", text)]
    return {
        "nonempty": bool(words),
        "complete_clause_edges": all(w.endswith((".", "!", "?"))
                                      for w in witness.left_words + witness.right_words),
        "repeated_units": len(words) != len(set(words)),
        "self_palindromic_units": any(letters(w) == letters(w)[::-1]
                                       for w in witness.left_words + witness.right_words),
        "catalogue_text": False,
        "posthoc_repair": False,
    }


def _frontier(report: Report) -> dict:
    states = sorted(report.reachable, key=repr)
    candidates = [s for s in states if s.residual]
    deepest = max(candidates, key=lambda s: len(s.residual), default=None)
    return {
        "reachable_states": len(report.reachable),
        "coaccessible_states": len(report.coaccessible),
        "intermediate_empty_closures_rejected": report.intermediate_empty_closures,
        "deepest_nonempty_residual": len(deepest.residual) if deepest else 0,
        "first_unsupported_frontier": (repr(deepest) if deepest else None),
        "next_grammar_operator": (
            "add one authored acknowledgement clause whose exposed reverse tape "
            "starts with the frontier debt, then re-run the same finite product"
        ),
    }


def run() -> dict:
    report = search(LEFT, RIGHT, max_states=5000, max_results=100,
                    reject_intermediate_closure=True)
    rows = []
    for witness in report.witnesses:
        text = render(witness)
        rows.append({"rendered": text, "length": len(letters(text)),
                     "provenance": {"source": "authored dialogue grammar",
                                     "left_words": list(witness.left_words),
                                     "right_words": list(witness.right_words),
                                     "phase_trace": list(witness.phase_trace)},
                     "audit": audit(text), "anti_shortcut": anti_shortcut(text, witness)})
    exact = [r for r in rows if r["audit"]["two_pointer_exact"]
             and r["length"] > 38 and not r["anti_shortcut"]["repeated_units"]]
    cycle_rows = []
    for pump in report.pumpable_cycles:
        for repetitions in range(1, 4):
            witness = materialize_pump(pump, repetitions)
            text = render(witness)
            cycle_rows.append({"rendered": text, "length": len(letters(text)),
                               "repetitions": repetitions,
                               "provenance": {"source": "authored dialogue grammar",
                                              "cycle_states": [repr(s) for s in pump.states]},
                               "audit": audit(text),
                               "anti_shortcut": anti_shortcut(text, witness)})
    return {
        "experiment_id": "natural-dialogue-open-cycle-20260922",
        "method": "finite product of two independently authored dialogue grammars with strict nonempty-residual cycle search",
        "grammar": {"left": {k: [e.word for e in v] for k, v in LEFT.items()},
                     "right": {k: [e.word for e in v] for k, v in RIGHT.items()}},
        "frontier": _frontier(report),
        "rendered_candidates": rows,
        "cycle_materializations": cycle_rows,
        "exact_candidates_gt38": exact,
        "novelty_preflight": {"status": "collision_with_complete_clause_atomic_lanes",
            "signature": "authored-dialogue|open-residual|recursive-product|complete-clause-edges",
            "catalogue_text": False, "fixed_tape": False, "posthoc_repair": False,
            "nested_palindrome_units": False},
        "reader_gate": {"status": "closed", "reason": "no new exact >38 output",
                         "human_evidence": "not collected"},
        "status": ("new exact output" if exact else
                    "rejected_as_complete_clause_atomic_probe"),
        "decision": ("This three-state failure occurs before dialogue or recurrence is tested. "
                     "Do not author a clause to copy the full residual; mine reusable fresh "
                     "lexicalized cycle transitions at an open debt instead."),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    data = run()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({"status": data["status"], "frontier": data["frontier"]}, sort_keys=True))
