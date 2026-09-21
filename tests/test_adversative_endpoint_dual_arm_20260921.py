import adversative_endpoint_dual_arm_20260921 as experiment


def test_adversative_controls_are_full_tape_and_content_disjoint():
    result = experiment.run()
    assert result["stats"]["controls"] == 2
    assert result["stats"]["max_matched_prefix"] == 1
    assert result["stats"]["accepted_exact"] == 0
    assert result["novelty_preflight"]["status"] == "passed"
    assert all(row["gates"]["content_disjoint"] for row in result["rendered_controls"])
    assert all(not row["accepted"] for row in result["rendered_controls"])

