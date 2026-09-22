import json
from pathlib import Path

from experiments.seed_np_cross_role_residual_continuation_20260922 import (
    CONTROL_SHA256,
    build_payload,
    independent_audit,
)


def test_control_and_strict_frontiers_are_preserved():
    result = build_payload()
    control = result["control"]
    assert control["audit"]["letters"] == 54
    assert control["audit"]["sha256_forward"] == CONTROL_SHA256
    assert control["audit"]["two_pointer_exact"]
    assert result["preserved_frontiers"]["568"]["letters"] == 568
    assert result["preserved_frontiers"]["666"]["letters"] == 666


def test_bounded_rows_persist_obstructions_without_promotion():
    result = build_payload()
    assert result["stats"]["exact_closures"] == 0
    assert result["stats"]["reader_certified"] == 0
    assert result["operator_change"]["changed_within_run"] is True
    for row in result["rows"]:
        assert row["length"] > 54
        assert row["independent_exact_audit"]["two_pointer_exact"] is False
        assert row["independent_exact_audit"] == independent_audit(row["rendered"])
        assert row["event_residual_obstruction"]["closed"] is False
        assert row["event_residual_obstruction"]["residual"]
        assert row["project_lexicon_gate"]["all_project_lexicon"] is True
        assert row["project_lexicon_gate"]["no_new_self_palindromic_word"] is True
        assert row["reader_status"] == "not_certified; no exact closure"
        assert row["promotion_status"] == "rejected_obstruction"


def test_prior_center_pair_is_rejected_evidence_only():
    result = build_payload()
    evidence = result["rejected_evidence"]
    assert evidence["commit"] == "a9d23763"
    assert "complete center pair" in evidence["reason"]
    assert all(row["provenance"]["prior_center_pair_reused"] is False for row in result["rows"])


def test_written_artifact_matches_generator():
    path = Path("runs/seed-np-cross-role-residual-continuation-20260922.json")
    if path.exists():
        written = json.loads(path.read_text())
        assert written["experiment_id"] == "seed-np-cross-role-residual-continuation-20260922"
