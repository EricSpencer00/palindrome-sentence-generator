"""Held-out agreement-carrying relatives with an optional passive frame.

This lane changes the grammar state, not the lexical bank.  Singular and
plural subjects select different finite verbs and auxiliaries, while a
relative clause may realize either an object gap (``that the writer reads``)
or a passive/participle frame (``that the writer has read``).  The shared-tape
support chart still propagates complete-parse support through mirrored
character domains before branching.
"""
from __future__ import annotations

import json
from pathlib import Path

from shared_tape_support_chart_20260920 import audit, calibration_grammar, chart, normalize, search

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs/agreement-passive-relative-chart-20260920.json"

LEXICON = {
    "DET_S": ("a", "the", "each", "every", "one"),
    "DET_P": ("some", "many", "several", "two", "three"),
    "PERSON_S": ("aide", "artist", "guard", "keeper", "poet", "sailor", "scribe", "writer"),
    "PERSON_P": ("artists", "guards", "keepers", "poets", "sailors", "scribes", "writers", "men"),
    "NAME_S": ("Diana", "Leon", "Mira", "Nora", "Rowan"),
    "PRON_S": ("I", "he", "she"),
    "PRON_P": ("they", "we"),
    "RELPRON": ("that", "who"),
    "VTRANS_S": ("admires", "carries", "charts", "copies", "guides", "hears", "keeps", "marks", "reads", "rips", "sees", "studies", "writes"),
    "VTRANS_P": ("admire", "carry", "chart", "copy", "guide", "hear", "keep", "mark", "read", "rip", "see", "study", "write"),
    "AUX_S": ("has", "is", "was"),
    "AUX_P": ("have", "are", "were"),
    "PART": ("carried", "charted", "copied", "guided", "heard", "kept", "marked", "read", "seen", "studied", "written"),
    "VINTR_S": ("arrives", "sings", "sleeps", "waits", "walks", "wonders"),
    "VINTR_P": ("arrive", "sing", "sleep", "wait", "walk", "wonder"),
    "TEXT": ("beacon", "chart", "charts", "garden", "harbor", "letter", "map", "maps", "memo", "memos", "note", "poem", "poems", "river", "story"),
    "NUM": ("one", "two", "three", "nine"),
    "PREP": ("after", "beside", "by", "from", "in", "near", "under", "with"),
    "CONJ": ("and", "while", "but"),
}

# Feature-bearing nonterminals prevent a singular relative subject from using
# a plural finite verb.  RELCLAUSE_* has two realizations: active object-gap
# and passive/participle with an agreeing auxiliary.
BINARY = (
    ("NPBASE_S", "DET_S", "PERSON_S"),
    ("NPBASE_P", "DET_P", "PERSON_P"),
    ("NPREL_S", "NPBASE_S", "REL_S"),
    ("NPREL_P", "NPBASE_P", "REL_P"),
    ("REL_S", "RELPRON", "RELCLAUSE_S"),
    ("REL_P", "RELPRON", "RELCLAUSE_P"),
    ("RELCLAUSE_S", "NPBASE_S", "VTRANS_S"),
    ("RELCLAUSE_P", "NPBASE_P", "VTRANS_P"),
    ("AUXPART_S", "AUX_S", "PART"),
    ("AUXPART_P", "AUX_P", "PART"),
    ("RELCLAUSE_S", "NPBASE_S", "AUXPART_S"),
    ("RELCLAUSE_P", "NPBASE_P", "AUXPART_P"),
    ("OBJ", "DET_S", "TEXT"),
    ("OBJ", "DET_P", "TEXT"),
    ("VP_S", "VTRANS_S", "OBJ"),
    ("VP_P", "VTRANS_P", "OBJ"),
    ("VP_S", "VTRANS_S", "PP"),
    ("VP_P", "VTRANS_P", "PP"),
    ("PP", "PREP", "NP"),
    ("C_S", "NP", "VP_S"),
    ("C_P", "NP", "VP_P"),
    ("CC", "C", "CONJTAIL"),
    ("CONJTAIL", "CONJ", "S"),
)
UNARY = (
    ("NP", "NPBASE_S"),
    ("NP", "NPBASE_P"),
    ("NPBASE_P", "PERSON_P"),  # bare plural relative subjects: ``writers read``
    ("NP", "NPREL_S"),
    ("NP", "NPREL_P"),
    ("NP", "NAME_S"),
    ("NP", "PRON_S"),
    ("NP", "PRON_P"),
    ("OBJ", "NP"),
    ("VP_S", "VINTR_S"),
    ("VP_P", "VINTR_P"),
    ("S", "C_S"),
    ("S", "C_P"),
    ("S", "CC"),
)


def run() -> dict:
    # Calibration is deliberately separate: it proves the solver can recover
    # the known 38-letter seed without promoting that seed as held-out output.
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
        "the poet that the writer reads studies a memo",
        "the poet that the writer has read studies a poem",
        "some poets who writers read carry two maps",
    ]
    control_rows = []
    for text in controls:
        tape = normalize(text)
        forest = chart([{c} for c in tape], LEXICON, BINARY, UNARY)
        control_rows.append({"rendered": text, "audit": audit(text),
                             "grammar_accepts": ("S", 0, len(tape)) in forest})
    rows = [row for result in held_out for row in result["exact_candidates"]]
    return {
        "experiment_id": "agreement-passive-relative-chart-20260920",
        "method": "root-supported shared-tape CFG with agreement-bearing active/passive relative clauses",
        "grammar": {
            "active_relative": "NPBASE_{sg|pl} RELPRON NPBASE_{sg|pl} VTRANS_{sg|pl} (object gap)",
            "passive_relative": "NPBASE_{sg|pl} RELPRON NPBASE_{sg|pl} AUX_{sg|pl} PART (object gap)",
            "matrix_agreement": "C_{sg|pl} -> NP VP_{sg|pl}",
            "recursive_sentence": "S -> C_sg | C_pl | C CONJ S",
            "independent_lexical_domains": {k: len(v) for k, v in LEXICON.items()},
        },
        "calibration": calibration,
        "calibration_recovered": recovered,
        "held_out_results": held_out,
        "exact_candidates": rows,
        "diagnostic_controls": control_rows,
        "reader_facing_candidates": [],
        "provenance": {
            "lexicon": "fresh authored role inventory with singular/plural verb and auxiliary classes",
            "fixed_tape_reversal": False,
            "per_search_rlaif": False,
            "independent_audits": ["outside-in character pointer", "forward/reverse SHA-256"],
            "reader_gate": "closed until exact, shortcut-clean output enters blinded intact/shuffled study",
        },
        "novelty_preflight": {
            "status": "passed",
            "signature": "shared-tape|root-support|agreement-features|active-passive-relative",
            "distinct_from": "object-relative bound-gap lane: finite and auxiliary/participle frames carry explicit subject-number state",
        },
        "next_construction": "add agreement-bearing relative-pronoun roles and a center-capable subject-relative frame; retain complete-parse support before lexical branching",
        "status": "exact candidate requires blinded readers" if rows else "no exact closure in held-out agreement/passive relative grammar",
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
