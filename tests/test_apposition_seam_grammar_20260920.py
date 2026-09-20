from apposition_seam_grammar_20260920 import audit, run


def test_apposition_lane_is_bounded_and_keeps_exact_gate_closed():
    result = run()
    assert result["stats"]["rendered_controls"] == 48
    assert result["stats"]["live_closed"] == 0
    assert result["stats"]["exact_gt38"] == 0
    assert all(row["provenance"]["independent_clause_authorship"] for row in result["rendered_controls"])


def test_apposition_audit_is_independent():
    result = audit("The keeper records a map.")
    assert result["exact"] is False
    assert result["sha_equal"] is False
