from experiments.center_seeded_semantic_expansion_20260920 import audit, center_gate, run


def test_nonpalindromic_center_is_rejected_before_expansion():
    assert center_gate("and")["palindromic"] is False
    assert center_gate("level")["palindromic"] is True


def test_center_lane_has_live_expansion_and_independent_audit():
    result = run(state_limit=10000)
    assert result["novelty_preflight"]["status"] == "passed"
    assert result["stats"]["transitions"] > 0
    assert len(result["center_gate"]["rejected"]) >= 3
    for row in result["rendered_candidates"]:
        assert audit(row["rendered"]) == row["audit"]
        assert row["provenance"]["center_selected_first"]
