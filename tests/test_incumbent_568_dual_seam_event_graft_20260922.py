import json

from experiments.incumbent_568_dual_seam_event_graft_20260922 import (
    FRONTIER,
    OUT,
    PARENT_SHA256,
    build_payload,
    independent_audit,
)


def test_dual_seam_event_graft_is_exact_and_longer_than_parent():
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
    assert row["paired_cursor_state"]["final_residual"] == ""
    assert row["seam_provenance"]["normalized_cuts"] == [108, 460]
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
