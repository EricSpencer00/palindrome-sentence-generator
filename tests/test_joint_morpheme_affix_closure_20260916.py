import json
from pathlib import Path

from experiments.joint_morpheme_affix_closure_20260916 import (
    EXPERIMENT_ID,
    MIN_LETTERS,
    STATE_SPACE_SIGNATURE,
    novelty_preflight,
    run,
)


def test_registry_preflight_is_self_registered_and_collision_free():
    result = novelty_preflight()
    assert result["status"] == "registered_self"
    assert result["exact_signature_collision"] == []
    assert result["artifact_collision"] == []


def test_joint_search_preserves_long_independent_failures_and_audits():
    result = run()
    assert result["experiment_id"] == EXPERIMENT_ID
    assert result["signature"] == STATE_SPACE_SIGNATURE
    assert result["stats"]["joint_pairs"] >= 10000
    assert result["stats"]["exact"] == 0
    assert len(result["failed_outputs"]) == 48
    for row in result["failed_outputs"]:
        assert row["letters"] >= MIN_LETTERS
        assert row["content_lemmas_disjoint"] is True
        assert row["independent_two_pointer"]["exact"] is False
        assert row["independent_hash_audit"]["exact"] is False
        assert row["repair_operator"]["operator"] == "first-mismatch-heldout-affix-swap"


def test_registry_points_to_source_and_frozen_run():
    root = Path(__file__).resolve().parents[1]
    registry = json.loads((root / "docs/experiment-novelty-registry.json").read_text())
    rows = [row for row in registry["entries"] if row["id"] == EXPERIMENT_ID]
    assert len(rows) == 1
    assert rows[0]["signature"] == STATE_SPACE_SIGNATURE
    assert (root / rows[0]["artifact"]).exists()
    assert all((root / path).exists() for path in rows[0]["run_artifacts"])
