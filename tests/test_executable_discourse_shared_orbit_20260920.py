import json
from pathlib import Path

from experiments.executable_discourse_shared_orbit_20260920 import audit, run


def test_joint_discourse_records_prelexical_semantic_prunes_and_controls():
    result = run()
    assert result["stats"]["semantic_prunes"] > 0
    assert result["stats"]["semantic_valid_pairs"] > 0
    assert result["stats"]["prose_controls"] == 3
    assert result["stats"]["exact_gt38"] == 0
    assert result["semantic_rejection_witnesses"][0]["lexical_spans_unresolved"] is True
    assert Path("runs/executable-discourse-shared-orbit-20260920.json").exists()


def test_independent_audit_detects_nonpalindrome():
    row = audit("The guard opened the gate, then the visitor entered the yard.")
    assert not row["pointer_exact"]
    assert row["sha256_forward"] != row["sha256_reverse"]
