"""Acceptance-facing checks for the latest parallel construction routes.

These tests keep the real gate visible: complete prose and independent audits
are preserved, but no route is promoted without an exact closure and readers.
"""
from __future__ import annotations

import json
import hashlib
import re
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
    assert report["candidate_count"] == 4426
    assert report["exact_count"] == 77
    assert report["mechanically_admitted_count"] == 0
    assert all(row["provenance"] != "unspecified" for row in report["rows"])
    assert all("brown_order_gain_vs_shuffle" in row["diagnostics_not_readability"]
               for row in report["rows"])
    assert len(report["route_summary"]) == 137
    assert max(row["max_letters"] for row in report["route_summary"]) == 1922


def test_latest_followup_lanes_are_retained_without_promoting_unreadable_text():
    seam = load("c3-seam-dp-semantic-pairs-20260916.json")
    manual = load("manual-bidirectional-scene-20260916.json")
    assert len(seam["candidates"]) == 9 and len(seam["repair"]) == 9
    assert seam["exact_count"] == 0 and seam["reader_eligible_count"] == 0
    assert len(manual["candidates"]) == 6
    assert manual["exact_count"] == 0 and manual["reader_eligible_count"] == 0
    for row in seam["candidates"] + seam["repair"]:
        tape = "".join(re.findall(r"[A-Za-z]", row["text"])).lower()
        assert row["audit"]["letters"] == len(tape)
        assert row["audit"]["exact"] is False
    for row in manual["candidates"]:
        tape = "".join(re.findall(r"[A-Za-z]", row["text"])).lower()
        assert row["normalized_length"] == len(tape)
        assert row["exact_independent_two_pointer"] is False


def test_aggregate_includes_nested_rendered_candidate_formats():
    report = load("parallel-luna-readability-diagnostics-20260916.json")
    by_source = {row["source_run"]: row for row in report["route_summary"]}
    assert by_source["runs/intact-prose-letter-repair-20260916.json"]["rows"] == 311
    assert by_source["runs/semantic-interleaving-frontback-20260916.json"]["rows"] == 15
    assert by_source["runs/regular-seam-grammar-20260916.json"]["rows"] == 5
    assert by_source["runs/human-guided-global-equation-editor-20260916.json"]["rows"] == 24
    assert by_source["runs/cumulative-boundary-profile-fixedpoint-20260916.json"]["rows"] == 647
    assert by_source["runs/joint-morpheme-affix-closure-20260916.json"]["rows"] == 48
    assert by_source["runs/equal-length-clause-collision-20260916.json"]["rows"] == 8
    assert by_source["runs/semantic-valency-boundary-csp-20260916.json"]["rows"] == 24
    assert by_source["runs/earley-finite-state-grammar-intersection-20260916.json"]["rows"] == 12
    assert by_source["runs/live-gpt2-character-decoder-preflight-20260916.json"]["rows"] == 20
    assert by_source["runs/dependency-tree-seam-solver-20260916.json"]["rows"] == 12
    assert by_source["runs/weighted-morphology-fst-lockstep-20260916.json"]["rows"] == 24
    assert by_source["runs/reader-first-discourse-scene-lattice-20260916.json"]["rows"] == 24
    assert by_source["runs/inflectional-clitic-boundary-csp-20260916.json"]["rows"] == 24
    assert by_source["runs/scalable-compositional-clause-grammar-20260916.json"]["rows"] == 36
    assert by_source["runs/exact-tape-semantic-slot-repair-20260916.json"]["rows"] == 24
    assert by_source["runs/typed-cfg-exact-tape-resegmentation-20260916.json"]["rows"] == 2
    assert by_source["runs/assumption-core-scene-solver-20260916.json"]["rows"] == 240
    assert by_source["runs/masked-character-scene-gibbs-20260916.json"]["rows"] == 3
    assert by_source["runs/min-cost-flow-clause-realizer-20260916.json"]["rows"] == 2
    assert by_source["runs/encoder-semantic-scene-pseudolikelihood-20260916.json"]["rows"] == 3
    assert by_source["runs/compound-derivational-scene-csp-20260916.json"]["rows"] == 48
    assert by_source["runs/paraphrase-graph-debt-paths-20260916.json"]["rows"] == 81
    assert by_source["runs/paraphrase-graph-debt-paths-20260916.json#repair"]["rows"] == 1
    assert by_source["runs/parse-tree-exact-cover-20260916.json#base"]["rows"] == 6
    assert by_source["runs/bilateral-semantic-cfg-20260916.json"]["rows"] == 1
    assert by_source["runs/human-scene-equation-frames-20260916.json"]["rows"] == 2
    assert by_source["runs/typed-edit-program-repair-20260916.json"]["rows"] == 18
    assert by_source["runs/event-graph-character-sat-20260916.json"]["rows"] == 81
    assert by_source["runs/phrase-equation-inventory-solver-20260916.json"]["rows"] == 279
    assert by_source["runs/syntax-stack-semantic-role-decoder-20260916.json"]["rows"] == 12
    assert by_source["runs/semantic-center-sat-20260916.json"]["rows"] == 9
    assert by_source["runs/reverse-segmentation-cfg-valency-20260916.json"]["rows"] == 3
    assert by_source["runs/dependency-mirror-pair-constructor-20260916.json"]["rows"] == 4
    assert by_source["runs/dependency-mirror-pair-repair-20260916.json"]["rows"] == 4
    assert by_source["runs/semantic-center-sat-repair-20260916.json"]["rows"] == 2
    assert by_source["runs/online-grammar-state-char-decoder-20260916.json"]["rows"] == 3
    assert by_source["runs/live-slot-equation-cfg-resegmentation-20260916.json"]["rows"] == 5
    assert by_source["runs/clause-growth-frame-repair-20260916.json"]["rows"] == 6
    assert by_source["runs/online-grammar-state-outer-frame-repair-20260916.json"]["rows"] == 2
    assert by_source["runs/reverse-lexicon-synthesis-20260916.json"]["rows"] == 4
    assert by_source["runs/centerout-grammar-boundary-dp-20260916.json"]["rows"] == 10
    assert by_source["runs/authored-clause-template-sat-20260916.json"]["rows"] == 13
    assert by_source["runs/authored-clause-template-sat-repair-20260916.json"]["rows"] == 1
    assert by_source["runs/centerout-grammar-boundary-repair-20260916.json"]["rows"] == 1
    assert by_source["runs/reverse-lexicon-inflection-repair-20260916.json"]["rows"] == 6
    assert by_source["runs/reverse-lexicon-boundary-seam-repair-20260916.json"]["rows"] == 4
    assert by_source["runs/centerout-grammar-boundary-repair2-20260916.json"]["rows"] == 1
    assert by_source["runs/authored-clause-template-sat-repair2-20260916.json"]["rows"] == 1
    assert by_source["runs/reverse-lexicon-shared-agreement-seams-20260916.json"]["rows"] == 3
    assert by_source["runs/centerout-grammar-boundary-repair3-20260916.json"]["rows"] == 1
    assert by_source["runs/authored-clause-template-sat-repair3-20260916.json"]["rows"] == 1
    assert by_source["runs/reverse-lexicon-centered-complement-20260916.json"]["rows"] == 1
    assert by_source["runs/centerout-grammar-boundary-repair4-20260916.json"]["rows"] == 1
    assert by_source["runs/authored-clause-template-sat-repair4-20260916.json"]["rows"] == 1
    assert by_source["runs/reverse-lexicon-typed-complement-frame-20260916.json"]["rows"] == 1
    assert by_source["runs/centerout-grammar-boundary-repair5-20260916.json"]["rows"] == 1
    assert by_source["runs/authored-clause-template-sat-repair5-20260916.json"]["rows"] == 1
    assert by_source["runs/reverse-lexicon-role-noun-boundary-20260916.json"]["rows"] == 1
    assert by_source["runs/centerout-paired-boundary-csp-20260916.json"]["rows"] == 1
    assert by_source["runs/authored-clause-template-sat-repair6-20260916.json"]["rows"] == 1
    assert by_source["runs/reverse-lexicon-adjacent-pp-boundary-20260916.json"]["rows"] == 1
    assert by_source["runs/centerout-paired-boundary-csp-verbframe-20260916.json"]["rows"] == 1
    assert by_source["runs/authored-clause-template-sat-repair7-20260916.json"]["rows"] == 1


def test_followup_preflights_are_explicit_and_generate_no_rows():
    for name in (
        "gpt2-reverse-rerank-preflight-20260916.json",
        "b3-corpus-weighted-reverse-preflight-20260916.json",
    ):
        run = load(name)
        assert run["status"] == "preflight_blocked"
        assert not run.get("rendered_candidates", [])
        assert run["pivot"]


def test_parallel_report_recomputes_every_tape_and_hash_independently():
    report = load("parallel-luna-readability-diagnostics-20260916.json")
    for row in report["rows"]:
        tape = "".join(re.findall(r"[A-Za-z]", row["rendered"])).lower()
        exact = bool(tape) and tape == tape[::-1]
        hashed = bool(tape) and hashlib.sha256(tape.encode()).hexdigest() == hashlib.sha256(tape[::-1].encode()).hexdigest()
        assert row["letters"] == len(tape)
        assert row["exact_letter_palindrome"] is exact
        assert row["independent_sha256_exact"] is hashed


def test_heldout_boundary_decoder_keeps_resegmentation_repairs_unadmitted():
    run = load("heldout-boundary-decoder-20260916.json")
    assert len(run["candidates"]) == 81
    assert len(run["repair"]) == 81
    assert run["exact_count"] == 0
    assert run["reader_eligible_count"] == 0
    assert all(row["audit"]["complete_sentence"] and not row["audit"]["reader_eligible"]
               for row in run["candidates"] + run["repair"])


def test_evidential_scene_planner_keeps_semantic_repairs_complete_but_unadmitted():
    run = load("evidential-scene-planner-20260916.json")
    assert len(run["base"]["candidates"]) == 4
    assert len(run["repair"]["candidates"]) == 4
    assert run["base"]["exact_count"] == 0
    assert run["repair"]["exact_count"] == 0
    assert all(row["complete_clauses"] == 2 and not row["reader_eligible"]
               for phase in ("base", "repair") for row in run[phase]["candidates"])


def test_residual_decoder_preserves_exact_but_unsegmentable_rejections():
    run = load("residual-lexical-decoder-20260916.json")
    assert run["survivors"] == []
    assert len(run["rejected"]) == 36
    assert run["exact_count"] == 0
    assert run["reader_eligible_count"] == 0
    assert all(row["audit"]["exact"] and not row["audit"]["all_words_lexicon"]
               and not row["audit"]["reader_eligible"] for row in run["rejected"])


def test_paragraph_paraphrase_obligation_keeps_intact_paragraphs_unadmitted():
    run = load("paragraph-paraphrase-obligation-20260916.json")
    assert len(run["candidates"]) == 3
    assert len(run["repair"]) == 3
    assert run["exact_count"] == 0
    assert run["reader_eligible_count"] == 0
    assert all(row["audit"]["complete_sentence"] and not row["audit"]["reader_eligible"]
               for row in run["candidates"] + run["repair"])


def test_heteropalindromic_clause_composer_keeps_fresh_pairs_unadmitted():
    run = load("heteropalindromic-clause-composer-20260916.json")
    assert len(run["candidates"]) == 12
    assert len(run["repair"]) == 12
    assert run["exact_count"] == 0
    assert run["reader_eligible_count"] == 0
    assert all(row["audit"]["complete_sentence"] and not row["audit"]["reader_eligible"]
               for row in run["candidates"] + run["repair"])


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


def test_direct_authoring_timeout_is_preserved_without_fabricated_text():
    run = load("direct-constrained-authoring-20260916.json")
    assert len(run["candidates"]) == 4
    assert run["exact_count"] == 0
    assert all(row["attempt"] is None and "TimeoutExpired" in row["error"]
               for row in run["candidates"])


def test_rhythmai_authoring_repair_is_a_real_failed_probe_not_a_candidate():
    run = load("rhythmai-authoring-probe-20260916.json")
    row = run["captured_output"]
    assert run["repair_of"] == "direct-constrained-authoring-20260916"
    assert row["letters"] == 30
    assert row["exact"] is False
    assert row["mechanically_admitted"] is False
    assert row["reader_eligible"] is False
    assert "exact_letter_palindrome" in [key for key, value in row["mechanical_checks"].items() if not value]


def test_lexical_chain_permutation_keeps_repair_probes_outside_gate():
    run = load("lexical-chain-palindrome-20260916.json")
    assert len(run["candidates"]) == 24
    assert len(run["repair"]) == 12
    assert run["exact_count"] == 0
    assert all(not row["audit"]["exact"] for row in run["candidates"] + run["repair"])


def test_microgrammar_lexical_debt_keeps_recursive_prose_and_repairs_unadmitted():
    run = load("microgrammar-lexical-debt-20260916.json")
    assert len(run["candidates"]) == 24
    assert len(run["repair"]) == 24
    assert run["exact_count"] == 0
    assert all(row["audit"]["letters"] >= 70 and not row["audit"]["exact"]
               for row in run["candidates"] + run["repair"])


def test_prosodic_and_induced_grammar_routes_keep_complete_prose_unadmitted():
    prosodic = load("prosodic-foot-scene-constructor-20260916.json")
    assert len(prosodic["base"]["candidates"]) == 54
    assert len(prosodic["repair"]["candidates"]) == 54
    assert all(row["complete_sentences"] and not row["audit"]["exact"]
               for phase in ("base", "repair") for row in prosodic[phase]["candidates"])
    induced = load("induced-grammar-reverse-decoder-20260916.json")
    assert len(induced["candidates"]) == 4
    assert len(induced["repair"]) == 4
    assert induced["exact_count"] == 0
    assert all(not row["audit"]["exact"] for row in induced["candidates"] + induced["repair"])


def test_semantic_frame_tape_solver_records_exact_but_unreadable_fragments():
    run = load("semantic-frame-tape-solver-20260916.json")
    assert len(run["candidates"]) == 81
    assert len(run["repair"]) == 81
    assert run["exact_count"] == 60
    exact_rows = [row for row in run["candidates"] + run["repair"] if row["audit"]["exact"]]
    assert exact_rows
    assert all(not row["audit"]["reader_eligible"] for row in exact_rows)
    assert any("nwadybpameulbaspeekesruneht" in row["text"] for row in exact_rows)


def test_conditional_embedding_route_keeps_complete_prose_and_repairs_unadmitted():
    run = load("conditional-embedding-solver-20260916.json")
    assert len(run["base"]["candidates"]) == 4
    assert len(run["repair"]["candidates"]) == 4
    assert run["base"]["exact_count"] == 0
    assert run["repair"]["exact_count"] == 0
    assert max(row["length_letters"] for phase in ("base", "repair")
               for row in run[phase]["candidates"]) == 89
    assert all(row["complete_clauses"] == 2 and not row["reader_eligible"]
               for phase in ("base", "repair") for row in run[phase]["candidates"])


def test_reported_speech_topology_preserves_embedded_propositions_and_repairs():
    run = load("reported-speech-topology-20260916.json")
    assert len(run["base"]["candidates"]) == 6
    assert len(run["repair"]["candidates"]) == 12
    assert run["base"]["exact_count"] == 0
    assert run["repair"]["exact_count"] == 0
    assert max(row["audit"]["letters"] for phase in ("base", "repair")
               for row in run[phase]["candidates"]) == 88
    assert all(row["complete_sentences"] and not row["reader_eligible"]
               for phase in ("base", "repair") for row in run[phase]["candidates"])


def test_nested_conditional_mutation_records_intact_prose_and_concrete_repairs():
    run = load("nested-conditional-mutation-20260916.json")
    assert len(run["candidates"]) == 3
    assert len(run["repair"]) == 6
    assert run["exact_count"] == 0
    assert run["reader_eligible_count"] == 0
    assert max(row["audit"]["letters"] for row in run["candidates"] + run["repair"]) == 50
    assert all(row["audit"]["complete_sentence"] and not row["audit"]["reader_eligible"]
               for row in run["candidates"] + run["repair"])


def test_simultaneous_phrase_beam_keeps_ngram_ranked_prose_outside_exact_gate():
    run = load("simultaneous-phrase-beam-20260916.json")
    assert len(run["candidates"]) == 9
    assert len(run["repair"]) == 9
    assert run["exact_count"] == 0
    assert run["reader_eligible_count"] == 0
    assert max(row["audit"]["letters"] for row in run["candidates"] + run["repair"]) == 72
    assert all(row["audit"]["complete_sentence"] and not row["audit"]["reader_eligible"]
               for row in run["candidates"] + run["repair"])


def test_discourse_connective_coupling_keeps_scene_repairs_complete_and_unadmitted():
    run = load("discourse-connective-coupled-20260916.json")
    assert len(run["candidates"]) == 6
    assert len(run["repair"]) == 6
    assert run["exact_count"] == 0
    assert run["reader_eligible_count"] == 0
    assert max(row["audit"]["letters"] for row in run["candidates"] + run["repair"]) == 57
    assert all(row["audit"]["complete_sentence"] and not row["audit"]["reader_eligible"]
               for row in run["candidates"] + run["repair"])


def test_topicalization_scope_route_preserves_roles_and_parenthetical_repairs():
    run = load("topicalization-scope-constructor-20260916.json")
    assert len(run["base"]["candidates"]) == 6
    assert len(run["repair"]["candidates"]) == 12
    assert run["base"]["exact_count"] == 0
    assert run["repair"]["exact_count"] == 0
    assert max(row["audit"]["letters"] for phase in ("base", "repair")
               for row in run[phase]["candidates"]) == 83
    assert all(row["complete_sentences"] and row["semantic_roles_preserved"]
               and not row["reader_eligible"]
               for phase in ("base", "repair") for row in run[phase]["candidates"])


def test_coupled_syntax_lexical_fallback_keeps_joint_repairs_unadmitted():
    run = load("coupled-syntax-lexical-authoring-20260916.json")
    assert len(run["candidates"]) == 9
    assert len(run["repair"]) == 9
    assert run["exact_count"] == 0
    assert run["reader_eligible_count"] == 0
    assert max(row["audit"]["letters"] for row in run["candidates"] + run["repair"]) == 50
    assert all(row["audit"]["complete_sentence"] and not row["audit"]["reader_eligible"]
               for row in run["candidates"] + run["repair"])


def test_reversible_semantic_wrappers_keep_fresh_centers_and_repairs_unadmitted():
    run = load("reversible-semantic-wrappers-20260916.json")
    assert len(run["candidates"]) == 9
    assert len(run["repair"]) == 9
    assert run["exact_count"] == 0
    assert run["reader_eligible_count"] == 0
    assert max(row["audit"]["letters"] for row in run["candidates"] + run["repair"]) == 62
    assert all(row["audit"]["complete_sentence"] and not row["audit"]["reader_eligible"]
               for row in run["candidates"] + run["repair"])


def test_comparative_modal_scene_route_keeps_measurement_prose_unadmitted():
    run = load("comparative-modal-scene-20260916.json")
    assert len(run["candidates"]) == 81
    assert len(run["repair"]) == 81
    assert run["exact_count"] == 0
    assert run["reader_eligible_count"] == 0
    assert all(row["audit"]["complete_sentence"] and not row["audit"]["reader_eligible"]
               for row in run["candidates"] + run["repair"])


def test_minimal_edit_fresh_centers_preserve_intact_prose_and_fail_exact_gate():
    run = load("minimal-edit-fresh-center-20260916.json")
    assert len(run["candidates"]) == 3
    assert len(run["repair"]) == 6
    assert run["exact_count"] == 0
    assert run["reader_eligible_count"] == 0
    assert all(row["audit"]["complete_sentence"] and not row["audit"]["reader_eligible"]
               for row in run["candidates"] + run["repair"])
    assert all("minimal-edit" in row.get("provenance", "") or row.get("operator") == "none"
               for row in run["candidates"] + run["repair"])


def test_scalar_evaluation_evidence_route_keeps_independent_semantics_unadmitted():
    run = load("scalar-evaluation-evidence-20260916.json")
    assert len(run["base"]["candidates"]) == 4
    assert len(run["repair"]["candidates"]) == 4
    assert run["base"]["exact_count"] == 0
    assert run["repair"]["exact_count"] == 0
    assert all(row["complete_clauses"] == 2 and not row["reader_eligible"]
               and row["semantic_topology"].startswith("event -> scalar evaluation")
               for phase in ("base", "repair") for row in run[phase]["candidates"])


def test_grammar_centerout_manual_route_preserves_fresh_prose_and_repair_trace():
    run = load("grammar-centerout-manual-20260916.json")
    assert len(run["candidates"]) == 3
    assert len(run["repair"]) == 3
    assert run["exact_count"] == 0
    assert run["reader_eligible_count"] == 0
    assert all(row["audit"]["complete_sentence"] and not row["audit"]["reader_eligible"]
               for row in run["candidates"] + run["repair"])
    assert all(row["provenance"].startswith("manual-common-word")
               for row in run["candidates"] + run["repair"])
