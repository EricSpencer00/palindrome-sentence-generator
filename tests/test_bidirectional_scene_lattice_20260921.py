from experiments.bidirectional_scene_lattice_20260921 import audit, run

def test_audit_is_independent_and_exact():
    row = audit("A man, a plan, a canal: Panama!")
    assert row["two_pointer_checked"] and row["pointer_exact"]
    assert row["sha256_forward"] == row["sha256_reverse"]

def test_joint_lattice_keeps_controls_and_residuals():
    result = run()
    assert result["novelty_preflight"]["status"] == "passed"
    assert result["stats"]["rendered_controls"] == 9
    assert result["stats"]["joint_states"] > 0
    assert result["residual_certificate"]
    assert all(r["provenance"]["selected_left_and_right_simultaneously"] for r in result["rendered_controls"])
    assert all(not r["provenance"]["finished_tape_reversal"] for r in result["rendered_controls"])
