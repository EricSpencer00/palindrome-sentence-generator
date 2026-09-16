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
    assert report["candidate_count"] == 838
    assert report["exact_count"] == 0
    assert report["mechanically_admitted_count"] == 0
    assert all(row["provenance"] != "unspecified" for row in report["rows"])
    assert all("brown_order_gain_vs_shuffle" in row["diagnostics_not_readability"]
               for row in report["rows"])
    assert len(report["route_summary"]) == 24
    assert max(row["max_letters"] for row in report["route_summary"]) == 1922


def test_seedless_and_reversible_clause_routes_keep_repairs_outside_reader_gate():
    seedless = load("seedless-semantic-cfg-bilateral-20260916.json")
    assert seedless["rendered_candidates"] == 10
    assert seedless["exact_candidates"] == 0
    assert all(row["complete_clauses"] >= 2 and not row["audit"]["exact"]
               for row in seedless["candidates"])
    family = load("whole-sentence-semordnilap-clauses-20260916.json")
    assert family["base"]["exact_count"] == 0
    assert family["repair"]["exact_count"] == 0
    assert all(not row["reader_eligible"] for phase in ("base", "repair")
               for row in family[phase]["probes"])


def test_semantic_mutation_keeps_complete_clauses_but_requires_exact_closure():
    run = load("semantic-mutation-residual-20260916.json")
    assert len(run["base"]["candidates"]) == 41
    assert len(run["repair"]["candidates"]) == 57
    assert run["base"]["exact_count"] == 0
    assert run["repair"]["exact_count"] == 0
    assert all(row["complete_clauses"] for phase in ("base", "repair")
               for row in run[phase]["candidates"])
    assert all(not row["reader_eligible"] for phase in ("base", "repair")
               for row in run[phase]["candidates"])


def test_fresh_rhetorical_plan_route_keeps_long_prose_outside_exact_gate():
    run = load("rhetorical-plan-lattice-20260916.json")
    assert run["base"]["exact_count"] == 0
    assert run["repair"]["exact_count"] == 0
    assert len(run["base"]["candidates"]) == 27
    assert len(run["repair"]["candidates"]) == 36
    assert max(row["audit"]["letters"] for phase in ("base", "repair")
               for row in run[phase]["candidates"]) == 172
    assert all(row["complete_prose"] and not row["reader_eligible"]
               for phase in ("base", "repair") for row in run[phase]["candidates"])


def test_inflectional_fst_and_clitic_repair_remain_complete_but_unadmitted():
    run = load("inflectional-fst-clitic-tape-20260916.json")
    assert len(run["base"]["candidates"]) == 64
    assert len(run["repair"]["candidates"]) == 32
    assert run["base"]["exact_count"] == 0
    assert run["repair"]["exact_count"] == 0
    assert all(row["complete_clauses"] and not row["reader_eligible"]
               for phase in ("base", "repair") for row in run[phase]["candidates"])


def test_induced_pcfg_records_derivation_search_without_fabricating_candidates():
    run = load("induced-pcfg-character-equation-20260916.json")
    assert run["base"]["derivations"] == 432
    assert run["repair"]["derivations"] == 1728
    assert run["base"]["exact_count"] == 0
    assert run["repair"]["exact_count"] == 0
    assert run["reader_eligible"] == []


def test_graph_to_prose_route_keeps_alternate_topologies_complete_and_unadmitted():
    run = load("graph-to-prose-path-20260916.json")
    assert len(run["base"]["candidates"]) == 9
    assert len(run["repair"]["candidates"]) == 16
    assert run["base"]["exact_count"] == 0
    assert run["repair"]["exact_count"] == 0
    assert all(row["complete_sentences"] and not row["reader_eligible"]
               for phase in ("base", "repair") for row in run[phase]["candidates"])


def test_voice_alternation_route_records_held_out_repair_without_exact_output():
    run = load("voice-alternation-residual-20260916.json")
    assert len(run["base"]["candidates"]) == 64
    assert len(run["repair"]["candidates"]) == 144
    assert run["base"]["exact_count"] == 0
    assert run["repair"]["exact_count"] == 0
    assert all(row["complete_clauses"] and not row["reader_eligible"]
               for phase in ("base", "repair") for row in run[phase]["candidates"])


def test_ccg_route_records_typed_derivations_and_reader_gate():
    run = load("ccg-semantic-solver-20260916.json")
    assert len(run["base"]["candidates"]) == 4
    assert len(run["repair"]["candidates"]) == 4
    assert run["base"]["exact_count"] == 0
    assert run["repair"]["exact_count"] == 0
    assert all(row["complete_clauses"] == 2 and not row["reader_eligible"]
               for phase in ("base", "repair") for row in run[phase]["candidates"])


def test_dependency_completion_csp_keeps_all_different_constraint_visible():
    run = load("dependency-completion-csp-20260916.json")
    assert len(run["base"]["candidates"]) == 81
    assert len(run["repair"]["candidates"]) == 81
    assert run["base"]["exact_count"] == 0
    assert run["repair"]["exact_count"] == 0
    assert all(row["complete_sentences"] and not row["reader_eligible"]
               for phase in ("base", "repair") for row in run[phase]["candidates"])


def test_lexical_word_equation_route_keeps_pos_choices_independent():
    run = load("lexical-word-equation-inventory-20260916.json")
    assert len(run["base"]["candidates"]) == 6
    assert len(run["repair"]["candidates"]) == 6
    assert run["base"]["exact_count"] == 0
    assert run["repair"]["exact_count"] == 0
    assert all(row["complete_clauses"] == 2 and row["all_different"]
               and not row["reader_eligible"]
               for phase in ("base", "repair") for row in run[phase]["candidates"])
