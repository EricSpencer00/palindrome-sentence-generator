"""Relative-complement extension of the typed residual scheduler.

This is a new grammar topology: a noun may carry one subject-gap relative
clause, while the same one-sided character-debt scheduler remains in force.
No exact survivor is repaired or wrapped after the search.
"""
from __future__ import annotations

import json
from pathlib import Path

from experiments.typed_residual_scheduler_20260920 import GRAMMAR as BASE_GRAMMAR
from experiments.typed_residual_scheduler_20260920 import load_lexicon, search

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/typed-relative-residual-scheduler-20260920.json"

GRAMMAR = dict(BASE_GRAMMAR)
GRAMMAR["CLAUSE"] = (("SUBJ", "V", "OBJ"), ("SUBJ", "V", "PP"), ("SUBJ", "ADV", "V", "OBJ"))
GRAMMAR["OBJ"] = (("N",), ("PROPN",), ("DET", "N"), ("DET", "ADJ", "N"),
                   ("NUM", "N"), ("DET", "N", "REL"), ("DET", "ADJ", "N", "REL"))
GRAMMAR["REL"] = (("RELPRON", "V", "N"), ("RELPRON", "V", "DET", "N"))


def run(*, lexicon_limit=70, max_words=18, max_nodes=500_000, beam_width=20_000):
    result = search(load_lexicon(limit=lexicon_limit, include_relative=True), grammar=GRAMMAR,
                    max_words=max_words, max_nodes=max_nodes, beam_width=beam_width)
    result["experiment_id"] = "typed-relative-residual-scheduler-20260920"
    result["provenance"] = {
        "method": "typed SVO/PP grammar plus one subject-gap relative complement with one-sided live residual scheduling",
        "lexicon_limit": lexicon_limit, "max_words": max_words,
        "max_nodes": max_nodes, "beam_width": beam_width,
        "relative_productions": ["OBJ -> DET N REL", "REL -> RELPRON V N", "REL -> RELPRON V DET N"],
        "one_sided_residual_scheduler": True, "nested_span_rejected": True,
        "candidate_reranking": False, "finished_tape_reversal": False,
        "post_hoc_repair": False, "reader_gate": "closed pending exact output and blinded intact-versus-shuffled reading",
        "next_construction": "if empty, change semantic topology rather than widening this relative beam",
    }
    return result


if __name__ == "__main__":
    result = run()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"]))
    for row in result["paths"][:20]:
        print(row["rendered"])
