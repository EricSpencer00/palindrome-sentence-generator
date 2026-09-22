"""Search complete clauses around one natural open-residual NP cycle.

Brown mining exposed two ordinary cyclic-reversal equations, ``no name`` /
``one man`` and ``no race`` / ``one car``.  This experiment uses each equation
once as a typed NP transition while independently selecting the surrounding
subject, past-tense transitive verb, object, and optional adjunct.  Exactness is
carried by the word-residual product before rendering; no language score or
post-hoc repair participates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import (
    REPEATABLE_FUNCTION_WORDS,
    mechanical_admission_checks,
    normalize_letters,
)
from llm_palindrome.dual_parse import letter_tape, word_residual_search


ID = "anchored-np-cycle-clause-search-20260922"
ANCHORS = {
    "name": {"residual": "name", "left": "no name", "right": "one man"},
    "race": {"residual": "race", "left": "no race", "right": "one car"},
}
SUBJECTS = (
    "the aide", "a clerk", "the guide", "a sailor", "the pilot",
    "a reader", "the ranger", "a mason",
)
VERBS = (
    "marked", "kept", "found", "carried", "recorded", "guarded",
    "remembered", "mapped", "named", "noted", "met", "saw",
)
OBJECTS = (
    "the map", "a memo", "the record", "a route", "the gate",
    "a letter", "the harbor", "a signal", "the trail", "a note",
)
ADJUNCTS = (
    "at dawn", "after rain", "near the harbor", "by the gate",
    "under the tree", "before noon",
)


def anchor_equation(anchor: dict) -> dict:
    residual = anchor["residual"]
    right_exposed = letter_tape(anchor["right"])[::-1]
    assert right_exposed.startswith(residual)
    remainder = right_exposed[len(residual):]
    left_exposed = letter_tape(anchor["left"])
    return {
        "start_debt": residual,
        "right_exposed": right_exposed,
        "mid_debt": remainder,
        "left_exposed": left_exposed,
        "end_debt": residual,
        "exact_open_cycle": left_exposed == remainder + residual,
    }


def plans(anchor: str, side: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    fixed = (anchor,)
    if side == "subject":
        return (
            ("subject:anchored-np", fixed),
            ("predicate:past-transitive", VERBS),
            ("object:ordinary-np", OBJECTS),
            ("adjunct", ADJUNCTS),
        )
    if side == "object":
        return (
            ("subject:ordinary-np", SUBJECTS),
            ("predicate:past-transitive", VERBS),
            ("object:anchored-np", fixed),
            ("adjunct", ADJUNCTS),
        )
    raise ValueError(side)


def pointer_audit(text: str) -> dict:
    tape = normalize_letters(text)
    mismatch = next((
        (index, tape[index], tape[-1 - index])
        for index in range(len(tape) // 2)
        if tape[index] != tape[-1 - index]
    ), None)
    return {
        "letters": len(tape),
        "exact": bool(tape) and mismatch is None,
        "first_mismatch": mismatch,
        "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest(),
    }


def _unique_content(left_words: tuple[str, ...],
                    right_words: tuple[str, ...]) -> bool:
    words = " ".join(left_words + right_words).casefold().split()
    content = [word for word in words if word not in REPEATABLE_FUNCTION_WORDS]
    return len(content) == len(set(content))


def run(*, max_states_per_pair: int = 250_000,
        max_results: int = 200) -> dict:
    rows, frontiers = [], []
    states = transitions = closures = intermediate = capped = 0
    for family, anchor in ANCHORS.items():
        equation = anchor_equation(anchor)
        assert equation["exact_open_cycle"]
        for left_position in ("subject", "object"):
            for right_position in ("subject", "object"):
                search = word_residual_search(
                    plans(anchor["left"], left_position),
                    plans(anchor["right"], right_position),
                    max_states=max_states_per_pair,
                    max_results=max_results - len(rows),
                    allow_partial=_unique_content,
                    reject_intermediate_closure=True,
                )
                states += search["states"]
                transitions += search["transitions"]
                intermediate += search["intermediate_closure_rejections"]
                capped += int(search["cap_reached"])
                for frontier in search["dead_frontiers"][:8]:
                    frontiers.append({
                        **frontier,
                        "family": family,
                        "left_anchor_position": left_position,
                        "right_anchor_position": right_position,
                    })
                for closure in search["results"]:
                    closures += 1
                    rendered = closure["left"].capitalize() + "; " + closure["right"] + "."
                    audit = pointer_audit(rendered)
                    checks = mechanical_admission_checks(
                        rendered, min_letters=39, max_letters=180
                    )
                    rows.append({
                        **closure,
                        "rendered": rendered,
                        "family": family,
                        "anchor_equation": equation,
                        "left_anchor_position": left_position,
                        "right_anchor_position": right_position,
                        "audit": audit,
                        "mechanical_admission": checks,
                        "mechanically_admitted": all(checks.values()),
                    })
    frontiers.sort(key=lambda row: (-row["matched_letters"],
                                    len(row["residual"])))
    admitted = [row for row in rows if row["mechanically_admitted"]]
    return {
        "experiment_id": ID,
        "method": "typed clause dual parse with one natural cyclic-reversal NP transition",
        "anchors": {name: {**value, "equation": anchor_equation(value)}
                    for name, value in ANCHORS.items()},
        "grammar": {
            "subjects": list(SUBJECTS), "verbs": list(VERBS),
            "objects": list(OBJECTS), "adjuncts": list(ADJUNCTS),
            "anchor_positions": ["subject", "object"],
        },
        "stats": {
            "grammar_pairs": 8,
            "states": states,
            "transitions": transitions,
            "capped_pairs": capped,
            "intermediate_closure_rejections": intermediate,
            "exact_closures": closures,
            "mechanically_admitted_gt38": len(admitted),
        },
        "exact_candidates": rows,
        "mechanically_admitted_candidates": admitted,
        "deepest_frontiers": frontiers[:40],
        "reader_packet": [],
        "provenance": {
            "anchor_discovery": "Brown cyclic-reversal diagnostic; common NP equations only",
            "surrounding_text": "fresh authored typed lexical domains",
            "borrowed_sentence": False,
            "anchor_used_once": True,
            "literal_cycle_pumping": False,
            "post_hoc_repair": False,
            "per_candidate_rlaif": False,
            "central_mechanical_admission": True,
            "readability_claim": "requires blinded human readers",
        },
        "status": (
            "exact admitted candidates require blinded study" if admitted else
            "no exact admitted closure; anchored NP cycle does not survive this clause grammar"
        ),
        "next_discriminator": (
            "If empty, retain the measured deepest frontier and change the grammar topology; "
            "do not widen these lexical banks or repeat the NP-anchor sweep."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-states-per-pair", type=int, default=250_000)
    parser.add_argument("--max-results", type=int, default=200)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    result = run(max_states_per_pair=args.max_states_per_pair,
                 max_results=args.max_results)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"stats": result["stats"],
                      "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
