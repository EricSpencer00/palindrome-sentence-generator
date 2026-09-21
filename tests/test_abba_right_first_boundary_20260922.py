from experiments.abba_right_first_boundary_20260922 import audit, run


def test_seed_audit_is_independent_and_exact():
    result = audit("An aide rips nine memos; some men inspire Diana.")
    assert result["two_pointer_exact"]
    assert result["sha256_forward"] == result["sha256_reverse_obligation"]


def test_right_first_lane_keeps_frames_controls_and_residuals():
    data = run()
    assert data["stats"]["right_first_frames"] == 3
    assert data["stats"]["left_controls"] == 9
    assert data["stats"]["closed_derivations"] == 0
    assert data["stats"]["exact_gt38"] == 0
    assert all(frame["selection_stage"] == "before-left-terminal-domain"
               for frame in data["right_first_frames"])
    assert all(cert["right_frame_selected_first"] for cert in data["residual_certificates"])
    assert all(cert["obligation_prefix"] for cert in data["residual_certificates"])
    assert data["novelty_preflight"]["status"] == "passed"
    assert all(row["audit"]["sha256_forward"] and
               row["audit"]["sha256_reverse_obligation"] for row in data["controls"])
