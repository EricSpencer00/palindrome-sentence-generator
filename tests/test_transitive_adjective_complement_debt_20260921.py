import experiments.transitive_adjective_complement_debt_20260921 as experiment


def test_transitive_control_does_not_claim_a_missing_right_arm():
    result = experiment.run()
    assert result["stats"]["controls"] == 81
    assert result["stats"]["paired_right_arm_controls"] == 0
    assert result["stats"]["accepted_exact"] == 0
    assert result["novelty_preflight"]["status"] == "passed"
    assert all(not row["accepted"] for row in result["rendered_controls"])
