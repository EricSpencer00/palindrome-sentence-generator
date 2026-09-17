import json
from pathlib import Path

from experiments.dependency_semantic_seam_csp_v2_20260917 import EXPERIMENT_ID, SIGNATURE, run
from llm_palindrome.admission import normalize_letters


def test_joint_seam_csp_exposes_complete_prose_and_independent_audit():
    report = run()
    assert report["experiment_id"] == EXPERIMENT_ID
    assert report["signature"] == SIGNATURE
    assert report["stats"]["pair_states"] > 1000
    assert report["stats"]["max_probe_letters"] > 80
    assert all(row["left_tree"]["readability"]["complete_clause"] for row in report["rendered_probes"])
    assert all(row["right_tree"]["readability"]["complete_clause"] for row in report["rendered_probes"])
    for row in report["rendered_probes"]:
        tape = normalize_letters(row["rendered"])
        assert row["audit"]["exact"] == (tape == tape[::-1])
        assert row["audit"]["pointer_sha256"]
        assert row["repair"]["operator"] if "repair" in row else True
    assert report["repair"]["candidate_count"] == 6
    assert all(row["right_tree"]["readability"]["complete_clause"] for row in report["repair_candidates"])
    assert all(row["provenance"]["heldout_place_domain"] for row in report["repair_candidates"])


def test_run_artifact_has_novelty_and_concrete_repair():
    report = run()
    assert report["novelty_preflight"]["performed_before_search"]
    assert report["provenance"]["catalogue_used"] is False
    assert report["repair"]["operator"]
    assert Path("runs/dependency-semantic-seam-csp-v2-20260917.json").exists()
