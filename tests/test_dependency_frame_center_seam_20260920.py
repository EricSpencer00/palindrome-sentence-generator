from experiments.dependency_frame_center_seam_20260920 import audit, run

def test_dependency_frame_lane_has_complete_controls_and_independent_audits():
    result = run()
    assert result["stats"]["states"] == 9
    assert result["stats"]["live_transitions"] > 0
    assert all(row["complete_prose"] for row in result["rendered_candidates"])
    assert all(row["audit"]["sha_equal_under_reverse"] == row["audit"]["exact"] for row in result["rendered_candidates"])
    assert result["novelty_preflight"]["duplicate_cartesian_sweep"] is False

def test_two_pointer_audit_rejects_nonpalindrome():
    assert audit("The keeper guards a bridge.")["exact"] is False
