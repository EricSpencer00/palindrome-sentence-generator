import json

from experiments.incumbent_568_dual_seam_event_graft_20260922 import (
    FRONTIER,
    OUT,
    PARENT_SHA256,
    build_payload,
    independent_audit,
)


def test_authored_event_graft_is_exact_and_longer_than_parent():
    payload = build_payload()
    row = payload["rows"][0]
    assert payload["preserved_frontier"] == list(FRONTIER)
    assert payload["stats"]["longest_letters"] == 622
    assert row["growth_over_parent"] == 54
    assert row["parent_sha256"] == PARENT_SHA256
    assert row["audit"]["two_pointer_exact"]
    assert row["audit"]["byte_pointer_exact"]
    assert row["audit"]["project_validator_exact"]
    assert row["independent_audit"]["two_pointer_exact"]
    assert row["independent_audit"]["sha_equal"]
    assert row["whole_graft_equation"]["equation_holds"]
    assert row["whole_graft_equation"]["incremental_residual_tracking"] is False
    assert row["whole_graft_equation"]["cursor_updates_computed_during_search"] is False
    assert row["whole_graft_equation"]["residual_owner"] is None
    assert row["seam_provenance"]["normalized_cuts"] == [108, 460]
    assert payload["novelty_preflight"]["novelty_status"].startswith("not novel")
    assert payload["novelty_preflight"]["same_cut_artifacts"]
    assert payload["novelty_preflight"]["prior_exact_event_hits"]
    assert "Aidan sees Mara." in row["rendered"]
    assert "Aram sees Nadia." in row["rendered"]


def test_artifact_reproduces_with_honest_global_checks():
    payload = build_payload()
    artifact = json.loads(OUT.read_text())
    assert artifact == payload
    row = artifact["rows"][0]
    assert independent_audit(row["rendered"]) == row["independent_audit"]
    assert not row["strict_global_checks"]["all_mechanical_checks"]
    assert row["strict_global_checks"]["human_certified"] is False
    assert "not an online residual search" in artifact["method"]
    assert "residual_before_close" not in artifact["rows"][0]["bounded_attempts"][0]
    assert "paired_cursor_state" not in row
