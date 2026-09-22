"""Held-out subject-relative and center-capable finite-clause chart.

Unlike the preceding object-relative lane, the head noun is the relative
subject: ``the poet who reads a memo``.  Active and auxiliary/participle forms
are represented without inserting an overt relative subject, which changes
the boundary and center topology while retaining agreement-bearing lexical
states and root-supported shared-tape propagation.
"""
from __future__ import annotations

import json
from pathlib import Path

from experiments.agreement_passive_relative_chart_20260920 import LEXICON
from experiments.shared_tape_support_chart_20260920 import audit, calibration_grammar, chart, normalize, search

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/subject-relative-center-chart-20260920.json"

BINARY = (
    ("NPBASE_S", "DET_S", "PERSON_S"),
    ("NPBASE_P", "DET_P", "PERSON_P"),
    ("NPREL_S", "NPBASE_S", "RELSUB_S"),
    ("NPREL_P", "NPBASE_P", "RELSUB_P"),
    ("RELSUB_S", "RELPRON", "RELSUBCLAUSE_S"),
    ("RELSUB_P", "RELPRON", "RELSUBCLAUSE_P"),
    ("RELSUBCLAUSE_S", "VTRANS_S", "OBJ"),
    ("RELSUBCLAUSE_P", "VTRANS_P", "OBJ"),
    ("RELSUBVP_S", "AUX_S", "PART"),
    ("RELSUBVP_P", "AUX_P", "PART"),
    ("RELSUBCLAUSE_S", "RELSUBVP_S", "OBJ"),
    ("RELSUBCLAUSE_P", "RELSUBVP_P", "OBJ"),
    ("OBJ", "DET_S", "TEXT"),
    ("OBJ", "DET_P", "TEXT"),
    ("VP_S", "VTRANS_S", "OBJ"),
    ("VP_P", "VTRANS_P", "OBJ"),
    ("C_S", "NP", "VP_S"),
    ("C_P", "NP", "VP_P"),
    ("CC", "C", "CONJTAIL"),
    ("CONJTAIL", "CONJ", "S"),
)
UNARY = (
    ("NP", "NPBASE_S"),
    ("NP", "NPBASE_P"),
    ("NP", "NPREL_S"),
    ("NP", "NPREL_P"),
    ("NP", "NAME_S"),
    ("NP", "PRON_S"),
    ("NP", "PRON_P"),
    ("NPBASE_P", "PERSON_P"),
    ("OBJ", "NP"),
    ("VP_S", "VINTR_S"),
    ("VP_P", "VINTR_P"),
    ("S", "C_S"),
    ("S", "C_P"),
    ("S", "CC"),
)


def run() -> dict:
    cal_lex, cal_binary, cal_unary = calibration_grammar()
    calibration = search(38, max_nodes=1000, lexicon=cal_lex,
                         binary=cal_binary, unary=cal_unary)
    seed = "an aide rips nine memos some men inspire diana".split()
    recovered = any([w.lower() for w in row["words"]] == seed
                    for row in calibration["exact_candidates"])
    held_out = [search(n, max_nodes=4000, lexicon=LEXICON,
                       binary=BINARY, unary=UNARY)
                for n in (39, 40, 44, 48, 52, 60, 72, 90, 100)]
    controls = [
        "the poet who reads a memo studies a map",
        "the poet who has read a poem studies a map",
        "some poets who carry two maps study a memo",
    ]
    control_rows = []
    for text in controls:
        tape = normalize(text)
        forest = chart([{c} for c in tape], LEXICON, BINARY, UNARY)
        control_rows.append({"rendered": text, "audit": audit(text),
                             "grammar_accepts": ("S", 0, len(tape)) in forest})
    rows = [row for result in held_out for row in result["exact_candidates"]]
    return {
        "experiment_id": "subject-relative-center-chart-20260920",
        "method": "root-supported shared-tape CFG with subject-relative head binding and center-capable finite clause",
        "grammar": {
            "subject_relative": "NPBASE_{sg|pl} RELPRON VTRANS_{sg|pl} OBJ (head is relative subject)",
            "subject_relative_auxiliary": "NPBASE_{sg|pl} RELPRON AUX_{sg|pl} PART OBJ",
            "matrix_agreement": "C_{sg|pl} -> NP VP_{sg|pl}",
            "recursive_sentence": "S -> C_sg | C_pl | C CONJ S",
        },
        "calibration": calibration,
        "calibration_recovered": recovered,
        "held_out_results": held_out,
        "exact_candidates": rows,
        "diagnostic_controls": control_rows,
        "reader_facing_candidates": [],
        "provenance": {
            "lexicon": "shared authored agreement inventory; new subject-relative productions only",
            "fixed_tape_reversal": False,
            "per_search_rlaif": False,
            "independent_audits": ["outside-in character pointer", "forward/reverse SHA-256"],
            "reader_gate": "closed until exact, shortcut-clean output enters blinded intact/shuffled study",
        },
        "novelty_preflight": {
            "status": "passed",
            "signature": "shared-tape|root-support|subject-relative-head-binding|center-capable-clause",
            "distinct_from": "agreement/passive object-relative lane: the head NP supplies the relative subject and the relative predicate consumes its object internally",
        },
        "next_construction": "add a subject-relative center bridge with an explicit complementizer and held-out semantic role frame; do not widen the existing lexical bank",
        "status": "exact candidate requires blinded readers" if rows else "no exact closure in held-out subject-relative grammar",
    }


if __name__ == "__main__":
    result = run()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "calibration": result["calibration"]["stats"],
        "calibration_recovered": result["calibration_recovered"],
        "controls": [(r["grammar_accepts"], r["audit"]["letters"]) for r in result["diagnostic_controls"]],
        "held_out": [(r["target_letters"], r["stats"]["status"], len(r["exact_candidates"]))
                     for r in result["held_out_results"]],
    }, sort_keys=True))
