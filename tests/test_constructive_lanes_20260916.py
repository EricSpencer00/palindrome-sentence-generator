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
    assert report["candidate_count"] == 4545
    assert report["exact_count"] == 78
    assert report["mechanically_admitted_count"] == 0
    by_source = {row["source_run"]: row for row in report["route_summary"]}
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
