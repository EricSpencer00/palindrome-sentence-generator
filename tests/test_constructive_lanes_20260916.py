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
    assert report["candidate_count"] == 4425
    assert report["exact_count"] == 77
    assert report["mechanically_admitted_count"] == 0
    by_source = {row["source_run"]: row for row in report["route_summary"]}
    assert by_source["runs/char-lm-tape-resegment-20260916.json"]["rows"] == 5
    assert by_source["runs/char-lm-multiclause-heldout-20260916.json"]["rows"] == 20
    assert by_source["runs/constructive-seam-morph-cfg-20260916.json"]["rows"] == 2
    assert by_source["runs/constructive-lanes-6-10-cross-audit-20260916.json"]["rows"] == 3
    assert by_source["runs/seam-feature-slot-repair-20260916.json"]["rows"] == 11
    assert by_source["runs/seam-feature-slot-repair-20260916.json#base"]["rows"] == 1
    assert by_source["runs/typed-boundary-resegment-shortwords-20260916.json"]["rows"] == 23


def test_seam_feature_repair_keeps_each_targeted_attempt_and_next_operator():
    run = json.loads((ROOT / "runs/seam-feature-slot-repair-20260916.json").read_text())
    assert run["novelty_preflight"]["status"] == "passed"
    assert len(run["attempts"]) == 11
    assert run["accepted"] == []
    assert all(row["audit"]["two_pointer"] is False for row in run["attempts"])
    assert all(row["audit"]["sha256"] for row in run["attempts"])
    assert "boundary resegmentation" in run["next_repair"]
