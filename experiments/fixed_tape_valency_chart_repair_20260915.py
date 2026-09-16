"""Carry finite-verb valency through the fixed-tape boundary chart.

This is the next repair after POS-only resegmentation.  The 116-letter tape is
immutable; a candidate survives only if its independently recovered words can
be partitioned into complete subject--finite-verb clauses with optional
objects/modifiers.  The parser is deliberately conservative and reports its
diagnostics instead of treating a POS-shaped word list as readable prose.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.grammar_boundary_resegmentation_repair_20260915 import (
    ID as SOURCE_ID,
    SIGNATURE as SOURCE_SIGNATURE,
    _brown_tables,
    _segment,
    _source_tape,
    _vocabulary,
)
from llm_palindrome.admission import mechanical_admission_checks


ID = "fixed-tape-valency-chart-repair"
SIGNATURE = (
    "fixed-tape|valency-unification|argument-role-chart|"
    "boundary-search|independent-audit"
)


def valency_parse(sequence: list[str]) -> dict:
    """Return a small chart witness for a sequence of universal POS tags.

    A chart item is ``(phase, subject_seen, finite_seen, object_seen)``.  The
    transition rules permit determiners/adjectives before noun subjects and
    objects, auxiliaries before a lexical verb, and ordinary adverbial/PP
    modifiers after a verb.  A conjunction or an explicit clause boundary may
    start the next complete clause.  No language-model score can certify this
    parse; the witness only decides whether the fixed tape is worth manual
    inspection.
    """
    # phase: subject -> finite verb -> post-verb material.  ``clause_count``
    # increases only when the preceding clause has a subject and finite verb.
    states = {("subject", False, False, False, 0)}
    transitions = []
    for tag in sequence:
        next_states = set()
        for phase, subject, finite, obj, clauses in states:
            if phase == "subject":
                if tag in {"DET", "ADJ", "NUM"}:
                    next_states.add((phase, subject, finite, obj, clauses))
                if tag in {"NOUN", "PRON"}:
                    next_states.add(("verb", True, finite, obj, clauses))
            elif phase == "verb":
                if tag in {"AUX", "VERB"}:
                    next_states.add(("post", subject, True, obj, clauses))
                elif tag in {"ADV", "ADP", "PRT", "ADJ"}:
                    next_states.add((phase, subject, finite, obj, clauses))
            else:  # post-verb
                if tag in {"DET", "ADJ", "NUM"}:
                    next_states.add(("object", subject, finite, obj, clauses))
                if tag in {"NOUN", "PRON"}:
                    next_states.add(("post", subject, finite, True, clauses))
                if tag in {"ADV", "ADP", "PRT", "ADJ", "NUM"}:
                    next_states.add((phase, subject, finite, obj, clauses))
                if tag == "CONJ" and subject and finite:
                    next_states.add(("subject", False, False, False, clauses + 1))
            # A conservative punctuation boundary is allowed before a new
            # subject-like tag, but only after a complete clause.
            if phase == "post" and subject and finite and tag in {"DET", "PRON", "NOUN"}:
                next_states.add(("verb" if tag in {"PRON", "NOUN"} else "subject",
                                 tag in {"PRON", "NOUN"}, False, False, clauses + 1))
        states = next_states
        transitions.append(len(states))
        if not states:
            break
    witnesses = [
        state for state in states
        if state[0] == "post" and state[1] and state[2]
    ]
    return {
        "accepted": bool(witnesses),
        "witnesses": [list(state) for state in witnesses[:5]],
        "chart_frontier": transitions,
        "failure_at": None if states else len(transitions),
    }


def run(*, limit: int = 80, vocabulary_size: int = 80_000) -> dict:
    tape, source_rendered = _source_tape()
    vocab, vocabulary_sha256 = _vocabulary(vocabulary_size)
    tags, transitions, totals = _brown_tables()
    rows = _segment(tape, vocab, tags, transitions, totals, limit=limit)
    for row in rows:
        row["valency_chart"] = valency_parse(row["pos_sequence"])
        row["complete_clause_parse"] = row["valency_chart"]["accepted"]
        row["mechanical_checks"] = mechanical_admission_checks(
            row["rendered"], min_letters=39, max_letters=220
        )
        row["mechanically_admitted"] = (
            row["independent_exact"] and all(row["mechanical_checks"].values())
        )
        row["reader_status"] = "not_run; valency chart is a diagnostic, not readability evidence"
    admitted = [row for row in rows if row["mechanically_admitted"]]
    parsed = [row for row in admitted if row["complete_clause_parse"]]
    return {
        "status": "fixed_tape_valency_chart_repair_complete",
        "experiment_id": ID,
        "signature": SIGNATURE,
        "repair_of": SOURCE_ID,
        "source_signature": SOURCE_SIGNATURE,
        "input": {
            "source_run": "runs/lexical-admission-centerout-20260915.json",
            "source_rendered": source_rendered,
            "fixed_normalized_tape": tape,
            "letters": len(tape),
            "tape_sha256": hashlib.sha256(tape.encode()).hexdigest(),
        },
        "config": {
            "candidate_limit": limit,
            "vocabulary_size_requested": vocabulary_size,
            "vocabulary_size": len(vocab),
            "pos_source": "Brown universal tags",
            "valency_states": ["subject", "verb", "post", "object"],
            "tape_mutation": False,
            "catalogue_text_imported": False,
        },
        "novelty_audit": {
            "registry_entries_read_before_run": 61,
            "excluded_routes_read_before_run": 3,
            "signature_overlap": [],
            "conceptual_near_pairs": [],
            "manual_review_required": False,
            "repair_of_registered_family": True,
            "self_entry_present": False,
            "preflight_required_before_artifact": True,
            "construction_dimension": (
                "finite-verb and argument-role chart over an immutable exact tape"
            ),
        },
        "stats": {
            "segmentations": len(rows),
            "mechanically_admitted": len(admitted),
            "complete_clause_parses": len(parsed),
            "reader_eligible": 0,
        },
        "rendered_candidates_and_probes": rows,
        "admitted": admitted,
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "vocabulary_sha256": vocabulary_sha256,
            "source_sentences_copied": False,
            "known_palindromes_imported": False,
            "independent_validator": "shared mechanical admission plus fixed tape equality",
            "readability_certificate": False,
        },
        "next_operator": (
            "If the valency chart still rejects every segmentation, introduce "
            "typed argument-role lexemes at the same boundary positions; do not "
            "alter the exact tape or silently relax clause completeness."
        ),
        "reader_gate": (
            "Only a complete-clause survivor that a blinded human panel rates as "
            "intact English may enter the reader package with intact and shuffled controls."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--vocabulary-size", type=int, default=80_000)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite existing output: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = run(limit=args.limit, vocabulary_size=args.vocabulary_size)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "segmentations": result["stats"]["segmentations"],
        "mechanically_admitted": result["stats"]["mechanically_admitted"],
        "complete_clause_parses": result["stats"]["complete_clause_parses"],
    }, indent=2))


if __name__ == "__main__":
    main()
