"""Regression checks for the ten-lane constructive continuation wave."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from llm_palindrome.admission import mechanical_admission_checks


ROOT = Path(__file__).parents[1]


def tape(text: str) -> str:
    return "".join(re.findall(r"[A-Za-z]", text)).lower()


def test_character_lm_resegmentation_keeps_all_authored_proposals_visible():
    run = json.loads((ROOT / "runs/char-lm-tape-resegment-20260916.json").read_text())
    assert len(run["candidates"]) == 5
    assert all(row["rendered"] for row in run["candidates"])
    assert run["stats"]["exact"] == 1
    exact = next(row for row in run["candidates"] if row["exact"])
    assert exact["rendered"] == "a man a plan a canal panama"
    assert tape(exact["rendered"]) == tape(exact["rendered"])[::-1]


def test_multiclause_character_lane_has_actual_over38_prose_and_no_closure():
    run = json.loads((ROOT / "runs/char-lm-multiclause-heldout-20260916.json").read_text())
    assert run["stats"] == {"pairs": 20, "eligible_over38": 20, "exact_over38": 0}
    assert all(row["rendered"] and row["letters"] > 38 for row in run["candidates"])
    assert all(not row["exact"] and not row["independent_exact_validation"]
               for row in run["candidates"])


def test_seam_morphology_cfg_exact_witness_is_rejected_by_quality_gates():
    run = json.loads((ROOT / "runs/constructive-seam-morph-cfg-20260916.json").read_text())
    candidate = run["candidate"]
    assert candidate == "Ava saw radar level civic; civic level radar was Ava"
    normalized = tape(candidate)
    assert len(normalized) == 42 and normalized == normalized[::-1]
    assert run["audit"]["sha256"] == hashlib.sha256(normalized.encode()).hexdigest()
    checks = mechanical_admission_checks(candidate, min_letters=39, max_letters=1000)
    assert checks["distinct_words"] is False
    assert checks["no_self_palindromic_word"] is False
    assert checks["not_word_order_symmetry"] is False


def test_scene_lattice_lanes_have_dual_audits_and_heldout_repairs():
    run = json.loads((ROOT / "runs/constructive-lanes-6-10-cross-audit-20260916.json").read_text())
    assert len(run["candidate_prose"]) == 3
    assert len(run["heldout_repairs"]) == 2
    assert run["exact_count"] == 0
    assert run["independent_exact_agreement"] is True
    for row in run["candidate_prose"]:
        assert row["rendered"] and row["audit"]["exact"] is False
        assert row["provenance"]
        assert row["next_repair"]["operator"]


def test_common_audit_includes_the_new_wave_without_reader_promotion():
    report = json.loads((ROOT / "runs/parallel-luna-readability-diagnostics-20260916.json").read_text())
    assert report["candidate_count"] == 5361
    assert report["exact_count"] == 84
    assert report["mechanically_admitted_count"] == 0
    by_source = {row["source_run"]: row for row in report["route_summary"]}
    assert by_source["runs/semantic-slot-attachment-repair-20260916-luna.json"]["rows"] == 2
    assert by_source["runs/finite-feature-center-grammar-20260916.json"]["rows"] == 3
    assert by_source["runs/joint-constituent-equation-scene-solver-20260916.json"]["rows"] == 9
    assert by_source["runs/word-internal-seam-equation-20260916.json"]["rows"] == 1
    assert by_source["runs/minimal-residual-grammar-20260916.json"]["rows"] == 4
    assert by_source["runs/semantic-relation-plan-solver-20260916.json"]["rows"] == 12
    assert by_source["runs/word-internal-seam-first-mismatch-repair-20260916.json"]["rows"] == 1
    assert by_source["runs/minimal-residual-grammar-repair-20260916.json"]["rows"] == 1
    assert by_source["runs/joint-constituent-equation-scene-repair-20260916.json"]["rows"] == 1
    assert by_source["runs/agreement-clitic-character-transducer-20260916.json"]["rows"] == 1
    assert by_source["runs/seed-benchmark-live-semantic-slot-expansion-20260916.json"]["rows"] == 1
    assert by_source["runs/semantic-valency-attachment-scene-lattice-20260916.json"]["rows"] == 6
    assert by_source["runs/agreement-clitic-outer-terminal-search-20260916.json"]["rows"] == 9
    assert by_source["runs/semantic-valency-clause-equation-solver-20260916.json"]["rows"] == 6
    assert by_source["runs/semantic-slot-exact-closure-frontier-20260916.json"]["rows"] == 1
    assert by_source["runs/outside-in-role-phrase-equation-20260916.json"]["rows"] == 1
    assert by_source["runs/cfg-earley-character-equation-forest-20260916.json"]["rows"] == 2
    assert by_source["runs/semantic-slot-first-residual-repair-20260916.json"]["rows"] == 6
    assert by_source["runs/char-lm-obligation-beam-20260916.json"]["rows"] == 64
    assert by_source["runs/center-free-clause-equation-ledger-20260916.json"]["rows"] == 1
    assert by_source["runs/inflectional-clitic-boundary-repair-20260916.json"]["rows"] == 10
    assert by_source["runs/dependency-seam-attachment-csp-20260916.json"]["rows"] == 64
    assert by_source["runs/center-residual-targeted-repair-20260917.json"]["rows"] == 8
    assert by_source["runs/dependency-role-first-residual-repair-20260917.json"]["rows"] == 5
    assert by_source["runs/char-lm-tape-resegment-20260916.json"]["rows"] == 5
    assert by_source["runs/char-lm-multiclause-heldout-20260916.json"]["rows"] == 20
    assert by_source["runs/constructive-seam-morph-cfg-20260916.json"]["rows"] == 2
    assert by_source["runs/constructive-lanes-6-10-cross-audit-20260916.json"]["rows"] == 3
    assert by_source["runs/seam-feature-slot-repair-20260916.json"]["rows"] == 11
    assert by_source["runs/seam-feature-slot-repair-20260916.json#base"]["rows"] == 1
    assert by_source["runs/typed-boundary-resegment-shortwords-20260916.json"]["rows"] == 23
    assert by_source["runs/fresh-typed-frame-live-seam-20260916.json"]["rows"] == 1
    assert by_source["runs/adjunct-boundary-targeted-repair-20260916.json"]["rows"] == 1
    assert by_source["runs/bidirectional-scene-decoder-20260916.json"]["rows"] == 1
    assert by_source["runs/semantic-slot-lattice-smt-20260916/run.json"]["rows"] == 108
    assert by_source["runs/clause-pair-csp-central-pivot-20260916/run.json"]["rows"] == 1
    assert by_source["runs/corpus-backed-reverse-segmentation-20260916.json"]["rows"] == 2
    assert by_source["runs/wordpair-graph-2026-09-16.json"]["rows"] == 1
    assert by_source["runs/paired-semantic-mutation-20260916.json"]["rows"] == 2
    assert by_source["runs/wordpair-graph-repair-2026-09-16.json"]["rows"] == 1
    assert by_source["runs/semantic-scene-repair-lane6-20260916.json"]["rows"] == 2
    assert by_source["runs/char-lm-grammar-2026-09-16.json"]["rows"] == 1
    assert by_source["runs/dependency-seam-csp-20260916.json"]["rows"] == 1
    assert by_source["runs/agreement-morphology-clitic-lane4-20260916.json"]["rows"] == 1
    assert by_source["runs/cfg-earley-joint-intersection-20260916.json"]["rows"] == 1
    assert by_source["runs/hand-authored-clause-breakthrough-2026-09-16.json"]["rows"] == 1
    assert by_source["runs/boundary-fst-resegment-2026-09-16.json"]["rows"] == 1
    assert by_source["runs/live-dependency-character-csp-20260916.json"]["rows"] == 1
    assert by_source["runs/semantic-residual-slot-lattice-20260916.json"]["rows"] == 1
    assert by_source["runs/outside-in-scene-grammar-csp-20260916.json"]["rows"] == 6
    assert by_source["runs/ten-clause-residual-equation-20260916.json"]["rows"] == 1
    assert by_source["runs/finite-automaton-clause-tapes-20260916.json"]["rows"] == 1
    assert by_source["runs/semordnilap-typed-clause-2026-09-16.json"]["rows"] == 2
    assert by_source["runs/two-sided-discourse-equation-20260916.json"]["rows"] == 1
    assert by_source["runs/endpoint-aware-bilateral-seam-20260916.json"]["rows"] == 6
    assert by_source["runs/fresh-scene-tape-cfg-resegmentation-20260916.json"]["rows"] == 2
    assert by_source["runs/exact-candidate-slot-repair-neighborhood-20260916.json"]["rows"] == 6
    assert by_source["runs/compositional-slot-boundary-dp-20260916.json"]["rows"] == 6
    assert by_source["runs/semantic-phrase-edge-graph-joiner-20260916.json"]["rows"] == 3
    assert by_source["runs/coupled-object-attachment-repair-20260916.json"]["rows"] == 4
    assert by_source["runs/semordnilap-role-clause-product-20260916.json"]["rows"] == 4
    assert by_source["runs/typed-reversible-clause-composer-20260916.json"]["rows"] == 2
    assert by_source["runs/seed-extension-frame-insertion-20260916.json"]["rows"] == 6
    assert by_source["runs/interrogative-relative-template-solver-20260916.json"]["rows"] == 2
    assert by_source["runs/live-tape-clause-terminal-decoder-20260916.json"]["rows"] == 6
    assert by_source["runs/bilateral-semantic-growth-grammar-20260916.json"]["rows"] == 3
    assert by_source["runs/scene-lattice-attachment-csp-20260916.json"]["rows"] == 2
    assert by_source["runs/boundary-shift-semordnilap-grammar-20260916.json"]["rows"] == 2
    assert by_source["runs/boundary-shift-scene-equation-lattice-20260916.json"]["rows"] == 2
    assert by_source["runs/semantic-boundary-macro-fsm-20260916.json"]["rows"] == 3
    assert by_source["runs/past-tense-dependency-transducer-20260916.json"]["rows"] == 6
    assert by_source["runs/reverse-tape-relative-resegment-20260916.json"]["rows"] == 2
    assert by_source["runs/scene-semordnilap-graph-20260916.json"]["rows"] == 2
    assert by_source["runs/centerout-open-word-seam-20260916.json"]["rows"] == 2
    assert by_source["runs/brown-phrase-pair-seam-20260916.json"]["rows"] == 6
    assert by_source["runs/bilateral-role-lattice-repair-20260916.json"]["rows"] == 4
    assert by_source["runs/feature-carrying-center-cfg-20260916.json"]["rows"] == 2
    assert by_source["runs/cross-boundary-phrase-block-grammar-20260916.json"]["rows"] == 3
    assert by_source["runs/typed-boundary-block-scene-20260916.json"]["rows"] == 16
    assert by_source["runs/cross-pos-semordnilap-scene-cfg-20260916.json"]["rows"] == 2
    assert by_source["runs/outside-in-heldout-scene-20260916.json"]["rows"] == 3
    assert by_source["runs/connected-scene-joint-resegment-20260916.json"]["rows"] == 4
    assert by_source["runs/valency-clitic-live-lexicalizer-20260916.json"]["rows"] == 2
    assert by_source["runs/scalable-outsidein-phrase-pair-20260916.json"]["rows"] == 3
    assert by_source["runs/fresh-seed-benchmark-seam-growth-20260916.json"]["rows"] == 16
    assert by_source["runs/reversible-phrase-pair-scene-search-20260916.json"]["rows"] == 2
    assert by_source["runs/scalable-outsidein-paired-terminal-repair-20260916.json"]["rows"] == 3
    assert by_source["runs/fresh-seam-heldout-joint-repair-20260916.json"]["rows"] == 1
    assert by_source["runs/reversible-phrase-pair-role-repair-20260916.json"]["rows"] == 2
    assert by_source["runs/scalable-outsidein-opposing-terminal-repair2-20260916.json"]["rows"] == 3
    assert by_source["runs/fresh-seam-attachment-followup-20260916.json"]["rows"] == 1
    assert by_source["runs/reversible-phrase-directional-adjunct-repair-20260916.json"]["rows"] == 2
    assert by_source["runs/fresh-seam-connector-followup-20260916.json"]["rows"] == 1
    assert by_source["runs/scalable-outsidein-single-edge-repair3-20260916.json"]["rows"] == 1
    assert by_source["runs/reversible-phrase-determiner-repair-20260916.json"]["rows"] == 1
    assert by_source["runs/live-cfg-character-chart-20260916.json"]["rows"] == 6
    assert by_source["runs/fresh-crossword-seam-csp-20260916.json"]["rows"] == 16
    assert by_source["runs/corpus-seam-fresh-scene-grammar-20260916.json"]["rows"] == 2
    assert by_source["runs/live-cfg-chart-terminal-edge-repair-20260916.json"]["rows"] == 1
    assert by_source["runs/fresh-crossword-seam-csp-followup-20260916.json"]["rows"] == 1
    assert by_source["runs/corpus-seam-reauthored-repair-20260916.json"]["rows"] == 1
    assert by_source["runs/bidirectional-phrase-pair-growth-20260916.json"]["rows"] == 2
    assert by_source["runs/centerout-museum-scene-lattice-20260916.json"]["rows"] == 4
    assert by_source["runs/morphology-crossword-transducer-20260916.json"]["rows"] == 2
    assert by_source["runs/bidirectional-phrase-pair-adjunct-repair-20260916.json"]["rows"] == 1
    assert by_source["runs/centerout-museum-heldout-repair-20260916.json"]["rows"] == 1
    assert by_source["runs/morphology-crossword-single-repair-20260916.json"]["rows"] == 1


def test_fresh_typed_frame_preserves_prose_and_live_obligation_evidence():
    run = json.loads((ROOT / "runs/fresh-typed-frame-live-seam-20260916.json").read_text())
    assert run["novelty_preflight"]["status"] == "passed"
    assert run["candidate"] == "The baker carries a letter near the quiet harbor"
    assert run["audit"]["two_pointer"] is False
    assert run["audit"]["sha256"]
    assert len(run["live_obligations"]) == 8
    assert run["provenance"]["fresh_domains"] is True


def test_adjunct_boundary_repair_preserves_frame_and_records_next_slot():
    run = json.loads((ROOT / "runs/adjunct-boundary-targeted-repair-20260916.json").read_text())
    assert run["novelty_preflight"]["status"] == "passed"
    assert run["candidate"] == "The baker carries a letter by the quiet harbor"
    assert run["provenance"]["frame_preserved"] is True
    assert run["audit"]["two_pointer"] is False
    assert "determiner slot" in run["next_repair"]


def test_clause_pair_csp_keeps_long_complete_prose_and_separate_audits():
    run = json.loads((ROOT / "runs/clause-pair-csp-central-pivot-20260916/run.json").read_text())
    assert run["letters"] == 113
    assert run["clauses"]["complete"] is True
    assert run["clauses"]["different"] is True
    assert run["audits"]["two_pointer"]["exact"] is False
    assert run["audits"]["independent_reverse_sha"]["exact"] is False
    assert run["audits"]["two_pointer"]["first_mismatch"]["index"] == 0
    assert run["provenance"]["candidate_count"] == 192


def test_joint_slot_lattice_records_all_pruned_states():
    run = json.loads((ROOT / "runs/semantic-slot-lattice-smt-20260916/run.json").read_text())
    assert run["states"] == 108
    assert run["pruned"] == 108
    assert len(run["rendered_candidates"]) == 108
    assert run["accepted"] == []
    assert all(row["checks"]["two_pointer"] is False for row in run["rendered_candidates"])


def test_bidirectional_scene_decoder_rejects_known_control_with_exact_audit():
    run = json.loads((ROOT / "runs/bidirectional-scene-decoder-20260916.json").read_text())
    assert run["candidate"]["normalized_length"] == 51
    assert run["candidate"]["two_pointer_exact"] is True
    assert run["candidate"]["independent_exact"] is True
    assert run["candidate"]["mechanically_admitted"] is False
    assert run["novelty_preflight"]["admitted"] is False


def test_reverse_segmentation_keeps_fresh_long_clauses_when_dp_fails():
    run = json.loads((ROOT / "runs/corpus-backed-reverse-segmentation-20260916.json").read_text())
    assert run["exact_count"] == 0
    assert len(run["candidates"]) == 2
    assert all(row["rendered"] and row["source_letters"] > 100 for row in run["candidates"])
    assert all(row["segmentation"] is None and row["exact"] is False for row in run["candidates"])
    assert all(row["provenance"] for row in run["candidates"])


def test_wordpair_graph_preserves_long_intact_frontier_without_closure():
    run = json.loads((ROOT / "runs/wordpair-graph-2026-09-16.json").read_text())
    row = run["candidates"][0]
    assert run["closures"] == 0
    assert row["letters"] == 164
    assert row["exact"] is False
    assert row["pos_valency_gate"] is True
    # The graph lane stores audit provenance as method descriptions; the
    # candidate's exact boolean is independently recomputed above.
    assert "independent tape" in run["audits"]["pointer"]
    assert "SHA-256" in run["audits"]["hash"]


def test_wordpair_graph_repair_keeps_long_fresh_scene_and_residual():
    run = json.loads((ROOT / "runs/wordpair-graph-repair-2026-09-16.json").read_text())
    row = run["candidate"]
    assert row["letters"] == 239
    assert row["exact"] is False and row["admitted"] is False
    assert row["pointer_audit"]["equal"] is False
    assert row["hash_audit"]["rendered"]
    assert run["next_repair"]


def test_semantic_scene_repair_keeps_two_fresh_over100_controls():
    run = json.loads((ROOT / "runs/semantic-scene-repair-lane6-20260916.json").read_text())
    assert run["stats"] == {"rendered": 2, "over_100": 2, "exact": 0, "admitted": 0}
    assert all(row["rendered"] and row["letters"] > 100 for row in run["rendered_candidates"])
    assert all(row["independent_ascii_exact"] is False for row in run["rendered_candidates"])
    assert all(row["two_pointer_mismatches"] for row in run["rendered_candidates"])
    assert run["next_repair"]["operator"]


def test_char_lm_grammar_lane_keeps_complete_frontier_and_audits():
    run = json.loads((ROOT / "runs/char-lm-grammar-2026-09-16.json").read_text())
    assert len(run["candidates"]) == 1
    row = run["candidates"][0]
    assert row["letters"] == 77 and row["exact"] is False
    assert row["pointer_audit"]["equal"] is False
    assert run["provenance"]
    assert run["next_repair"]


def test_dependency_seam_csp_lane_keeps_typed_scene_and_residual():
    run = json.loads((ROOT / "runs/dependency-seam-csp-20260916.json").read_text())
    assert run["exact_count"] == 0 and run["admitted_count"] == 0
    row = run["candidates"][0]
    assert row["letters"] == 105 and row["exact"] is False
    assert row["novelty_preflight"]["catalogue_match"] is False
    assert run["next_repair"]


def test_agreement_morphology_lane_keeps_long_clitic_scene():
    run = json.loads((ROOT / "runs/agreement-morphology-clitic-lane4-20260916.json").read_text())
    assert run["stats"] == {"rendered": 1, "over_100": True, "exact": False, "admitted": False}
    row = run["rendered_candidates"][0]
    assert row["letters"] > 100 and row["independent_ascii_exact"] is False
    assert run["transducer"]["clitic_policy"]
    assert run["next_repair"]["operator"]


def test_cfg_earley_joint_lane_keeps_one_fresh_long_scene():
    run = json.loads((ROOT / "runs/cfg-earley-joint-intersection-20260916.json").read_text())
    assert run["stats"] == {"rendered": 1, "over_100": 1, "exact": 0}
    row = run["rendered_candidates"][0]
    assert row["letters"] == 140 and row["exact"] is False
    assert row["independent_hash"]
    assert run["novelty_preflight"]["catalogue_text_imported"] is False
    assert run["next_repair"]["operator"]


def test_hand_authored_clause_breakthrough_keeps_intact_308_letter_frontier():
    run = json.loads((ROOT / "runs/hand-authored-clause-breakthrough-2026-09-16.json").read_text())
    row = run["candidate"]
    assert row["letters"] == 308
    assert row["exact"] is False and row["admitted"] is False
    assert row["pointer_audit"]["exact"] is False
    assert row["hash_equal"] is False
    assert row["mechanical_checks"]["distinct_words"] is True
    assert run["provenance"]["all_different_content_words"] is True
    assert run["next_repair"]


def test_boundary_fst_resegment_keeps_long_prose_and_boundary_repair():
    run = json.loads((ROOT / "runs/boundary-fst-resegment-2026-09-16.json").read_text())
    row = run["candidate"]
    assert row["letters"] == 270 and row["exact"] is False and row["admitted"] is False
    assert row["pointer_audit"]["exact"] is False
    assert row["forward_hash"] != row["reverse_hash"]
    assert run["provenance"]["inventory"] == "fresh authored clauses"
    assert run["next_repair"]


def test_live_dependency_character_csp_keeps_typed_scene_and_equations():
    run = json.loads((ROOT / "runs/live-dependency-character-csp-20260916.json").read_text())
    row = run["candidates"][0]
    assert row["letters"] == 127 and row["exact"] is False and row["admitted"] is False
    assert row["two_pointer_exact"] is False
    assert row["direct_hash"] != row["reverse_hash"]
    assert row["checks"]["length_band"] is True
    assert row["provenance"]["fresh_complete_scene"] is True
    assert run["next_repair"]


def test_semantic_residual_slot_lattice_keeps_agreement_locked_scene():
    run = json.loads((ROOT / "runs/semantic-residual-slot-lattice-20260916.json").read_text())
    row = run["rendered_candidates"][0]
    assert row["letters"] == 122 and row["exact"] is False
    assert row["two_pointer"] is False
    assert row["forward_hash"] != row["reverse_hash"]
    assert row["checks"]["distinct_words"] is True
    assert run["coupling"]["agreement"] == "third-person singular present"
    assert run["next_repair"]["operator"]


def test_outside_in_scene_csp_filters_repeated_frames_and_keeps_fresh_prose():
    run = json.loads((ROOT / "runs/outside-in-scene-grammar-csp-20260916.json").read_text())
    assert len(run["candidates"]) == 6
    assert run["exact_count"] == 0 and run["admitted_count"] == 0
    assert all(row["letters"] >= 100 for row in run["candidates"])
    assert all(row["exact"] is False and row["two_pointer"] is False for row in run["candidates"])
    assert all(row["hash_equal"] is False for row in run["candidates"])
    assert all(row["checks"]["distinct_words"] for row in run["candidates"])
    assert all(row["provenance"]["all_different_content_words"] for row in run["candidates"])
    assert run["next_repair"]


def test_ten_clause_residual_equation_keeps_long_scene_and_joint_repair():
    run = json.loads((ROOT / "runs/ten-clause-residual-equation-20260916.json").read_text())
    row = run["rendered_candidates"][0]
    assert row["letters"] == 355 and row["exact"] is False
    assert row["two_pointer_exact"] is False
    assert row["forward_hash"] != row["reverse_hash"]
    assert row["mechanical_checks"]["distinct_words"] is True
    assert run["solver"]["all_different_content"] is True
    assert run["next_repair"]["operator"]


def test_finite_automaton_clause_lane_keeps_scalable_state_and_exact_trace():
    run = json.loads((ROOT / "runs/finite-automaton-clause-tapes-20260916.json").read_text())
    row = run["candidates"][0]
    assert row["letters"] == 109 and row["exact"] is False and row["admitted"] is False
    assert row["two_pointer"] is False and row["hash_equal"] is False
    assert row["checks"]["length_band"] is True
    assert row["provenance"]["fresh_complete_clauses"] is True
    assert run["next_repair"]


def test_semordnilap_typed_lane_keeps_short_control_and_100_letter_prose():
    run = json.loads((ROOT / "runs/semordnilap-typed-clause-2026-09-16.json").read_text())
    witness, near = run["candidates"]
    assert witness["letters"] == 16 and witness["exact"] is True and witness["admitted"] is False
    assert near["letters"] == 100 and near["exact"] is False
    assert near["hash_equal"] is False
    assert near["mechanical_checks"]["distinct_words"] is True
    assert run["provenance"]["distinct_content"] is True
    assert run["next_repair"]


def test_two_sided_discourse_equation_keeps_distinct_complete_clauses():
    run = json.loads((ROOT / "runs/two-sided-discourse-equation-20260916.json").read_text())
    row = run["rendered_candidates"][0]
    assert row["letters"] == 156 and row["exact"] is False
    assert row["two_pointer"] is False
    assert row["forward_hash"] != row["reverse_hash"]
    assert row["mechanical"]["distinct_words"] is True
    assert row["mechanical"]["no_repeated_nontrivial_unit"] is True
    assert run["equation_solver"]["content_words_all_different"] is True
    assert run["next_repair"]


def test_paired_semantic_mutation_retains_fresh_controls_and_mismatch_trace():
    run = json.loads((ROOT / "runs/paired-semantic-mutation-20260916.json").read_text())
    assert run["stats"]["rendered"] == 2
    assert run["stats"]["exact"] == 0
    assert all(row["coherent_scene_slots"] for row in run["rendered_candidates"])
    assert all(row["independent_ascii_exact"] is False for row in run["rendered_candidates"])
    assert all(row["two_pointer_mismatches"] for row in run["rendered_candidates"])


def test_seam_feature_repair_keeps_each_targeted_attempt_and_next_operator():
    run = json.loads((ROOT / "runs/seam-feature-slot-repair-20260916.json").read_text())
    assert run["novelty_preflight"]["status"] == "passed"
    assert len(run["attempts"]) == 11
    assert run["accepted"] == []
    assert all(row["audit"]["two_pointer"] is False for row in run["attempts"])
    assert all(row["audit"]["sha256"] for row in run["attempts"])
    assert "boundary resegmentation" in run["next_repair"]
