from experiments.brown_pos_template_intersection_20260917 import norm, pointer_audit, residual_dp, run

def test_direction_and_seed_control():
    assert residual_dp("level", "level")["closed"]
    assert not residual_dp("abc", "abd")["closed"]

def test_run_has_controls_and_audits():
    out = run()
    assert out["novelty_preflight"]["passed"]
    assert out["stats"]["novel_attempts"] == 1
    assert out["stats"]["over_38"] >= 1
    for row in out["rows"]:
        assert row["independent_audit_agreement"]
        assert "repair_operator" in row
