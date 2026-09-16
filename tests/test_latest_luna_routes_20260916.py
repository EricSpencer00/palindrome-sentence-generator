"""Acceptance-facing checks for the latest parallel construction routes.

These tests keep the real gate visible: complete prose and independent audits
are preserved, but no route is promoted without an exact closure and readers.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def load(name: str) -> dict:
    return json.loads((ROOT / "runs" / name).read_text())


def test_recursive_grammar_route_preserves_long_complete_probes_without_claiming_exactness():
    run = load("recursive-grammar-residual-dp-20260916.json")
    assert run["rendered_candidates"] == 40
    assert run["exact_candidates"] == 0
    assert run["reader_eligible"] == []
    assert max(row["independent_audit"]["letters"] for row in run["candidates"]) == 1922
    assert all(row["left_complete_clauses"] == row["right_complete_clauses"] for row in run["candidates"])
    assert all(not row["independent_audit"]["exact"] for row in run["candidates"])


def test_dialogue_repair_keeps_speech_act_and_reader_gate_closed():
    base = load("dialogue-speech-act-residual-20260916.json")
    repair = load("dialogue-speech-act-residual-repair-20260916.json")
    assert base["stats"]["exact"] == 0
    assert base["stats"]["complete_prose"] == base["stats"]["compatible_pairs"]
    assert repair["stats"]["repair_trials"] == 36
    assert repair["stats"]["complete_trials"] == 36
    assert repair["stats"]["exact_over_38"] == 0
    assert all(row["audit"]["complete_left"] and row["audit"]["complete_right"]
               for row in base["rows"])
    assert all(row["complete_left"] and row["complete_right"] for row in repair["rows"])


def test_brown_lattice_records_zero_closure_without_fabricating_text():
    run = load("brown-attested-residual-lattice-20260916.json")
    assert run["source_sentence_count"] == 57340
    assert run["exact_closures"] == 0
    assert run["readable_over_38"] == 0
    assert run["repair_adjacent_span_candidates"] == 0
    assert run["displayed"] == []


def test_authored_boundary_search_keeps_all_probes_complete_and_unadmitted():
    run = load("reverse-transition-svo-authored-search-20260916.json")
    assert run["stats"]["exact"] == 0
    assert run["stats"]["mechanically_admitted"] == 0
    assert run["stats"]["rendered_probes"] == 3
    assert all(row["checks"]["word_form"] and row["checks"]["lexicon_words"]
               for row in run["rendered_probes"])


def test_parallel_readability_report_is_diagnostic_and_keeps_provenance():
    report = load("parallel-luna-readability-diagnostics-20260916.json")
    assert report["status"] == "diagnostic_not_human_readability_result"
    assert report["candidate_count"] == 166
    assert report["exact_count"] == 0
    assert report["mechanically_admitted_count"] == 0
    assert all(row["provenance"] != "unspecified" for row in report["rows"])
    assert all("brown_order_gain_vs_shuffle" in row["diagnostics_not_readability"]
               for row in report["rows"])


def test_seedless_and_reversible_clause_routes_keep_repairs_outside_reader_gate():
    seedless = load("seedless-semantic-cfg-bilateral-20260916.json")
    assert seedless["rendered_candidates"] == 10
    assert seedless["exact_candidates"] == 0
    assert all(row["complete_clauses"] == 2 and not row["audit"]["exact"]
               for row in seedless["candidates"])
    family = load("whole-sentence-semordnilap-clauses-20260916.json")
    assert family["base"]["exact_count"] == 0
    assert family["repair"]["exact_count"] == 0
    assert all(not row["reader_eligible"] for phase in ("base", "repair")
               for row in family[phase]["probes"])
