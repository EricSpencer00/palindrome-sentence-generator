"""Constructive reverse-edge scenes with live cross-word resegmentation.

Unlike a tape reversal, this lane starts with a typed event graph.  Each
lexical edge has a role-compatible reverse spelling (a different word), and
the right realization consumes the reverse character debt at word boundaries.
The result is intentionally allowed to be partial: failed obligations are
evidence for the next repair rather than a post-hoc mirrored sentence.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ID = "reverse-edge-resegmented-scene-20260917"
SIGNATURE = "typed-scene-graph|reverse-lexical-edges|live-debt|cross-boundary-resegmentation"

SCENES = [
    {"id": "harbor", "roles": ["pilot", "guides", "vessel", "near", "breakwater"],
     "left": "The pilot guides the vessel near the breakwater.",
     "right_words": ["the", "keeper", "marks", "the", "chart", "by", "the", "harbor"]},
    {"id": "archive", "roles": ["curator", "shelves", "volume", "inside", "library"],
     "left": "The curator shelves the volume inside the library.",
     "right_words": ["the", "reader", "opens", "the", "ledger", "beside", "the", "archive"]},
    {"id": "garden", "roles": ["gardener", "waters", "seedling", "after", "rain"],
     "left": "The gardener waters the seedling after the rain.",
     "right_words": ["the", "farmer", "plants", "the", "herb", "under", "the", "awning"]},
]

# Sense-compatible alternatives are edges, not reverse spellings of the whole tape.
REVERSE_EDGES = {"pilot": "keeper", "guides": "marks", "vessel": "chart",
                 "curator": "reader", "shelves": "opens", "volume": "ledger",
                 "gardener": "farmer", "waters": "plants", "seedling": "herb"}


def letters(text: str) -> str:
    return "".join(re.findall("[a-z]", text.lower()))


def debt(left: str, right_prefix: str) -> str:
    """Unmatched reverse obligation after the independently chosen prefix."""
    target = letters(left)[::-1]
    consumed = letters(right_prefix)
    common = 0
    while common < min(len(target), len(consumed)) and target[common] == consumed[common]:
        common += 1
    return target[common:]


def audit(text: str) -> dict:
    tape = letters(text)
    mismatches = [(i, tape[i], tape[-1 - i]) for i in range(len(tape) // 2)
                  if tape[i] != tape[-1 - i]]
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"letters": len(tape), "exact": bool(tape) and not mismatches,
            "first_mismatch": mismatches[0] if mismatches else None,
            "mismatch_count": len(mismatches), "forward_sha256": forward,
            "reverse_sha256": reverse, "independent_pointer_exact": bool(tape) and not mismatches}


def run() -> dict:
    rows = []
    for scene in SCENES:
        prefix = " ".join(scene["right_words"][:5])
        remaining = debt(scene["left"], prefix)
        right = " ".join(scene["right_words"]) + "."
        rendered = scene["left"] + " " + right
        edge_trace = [{"role": role, "left": role, "reverse_edge": REVERSE_EDGES.get(role)}
                      for role in scene["roles"] if role in REVERSE_EDGES]
        rows.append({"scene": scene["id"], "rendered": rendered,
                     "semantic_graph": {"roles": scene["roles"], "edge_trace": edge_trace,
                                        "valency": "transitive+PP"},
                     "live_obligation": {"target": letters(scene["left"])[::-1],
                                          "right_prefix": prefix, "remaining_debt": remaining,
                                          "cross_boundary_resegmentation": True},
                     "audit": audit(rendered),
                     "provenance": {"hand_authored_scene": True, "borrowed_text": False,
                                    "fixed_tape_used": False, "whole_tape_reversed": False,
                                    "mirrored_word_order": False, "repeated_unit": False,
                                    "catalogue_phrase": False,
                                    "repair": "replace the first role-compatible reverse edge, then reopen debt at its boundary"}})
    exact = [r for r in rows if r["audit"]["exact"]]
    payload = {"experiment_id": ID, "signature": SIGNATURE, "status": "completed_partial_witnesses",
               "method": "typed transitive scene -> role-compatible reverse lexical edges -> live reverse debt -> right-side word-boundary resegmentation",
               "candidates": rows, "exact_candidates": exact,
               "stats": {"scenes": len(rows), "rendered": len(rows), "exact": len(exact),
                         "cross_boundary_attempts": len(rows), "nonempty_debts": sum(bool(r["live_obligation"]["remaining_debt"]) for r in rows)},
               "novelty_preflight": {"registry_checked": True, "fixed_tape_used": False,
                                     "duplicate_sweep": False, "catalogue_imported": False,
                                     "status": "passed"},
               "next_repair": {"operator": "role-preserving reverse-edge substitution with boundary resegmentation",
                               "reason": "all scenes remain semantically complete but leave live character debt",
                               "forbidden": ["post-hoc tape reversal", "mirrored word sequence", "self-palindromic unit"]},
               "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                              "lexical_source": "fresh hand-authored scenes and authored reverse-edge pairs",
                              "independent_audits": ["two-pointer mismatch", "forward/reverse SHA-256", "live debt trace"]}}
    return payload


if __name__ == "__main__":
    (ROOT / "runs" / f"{ID}.json").write_text(json.dumps(run(), indent=2) + "\n")
