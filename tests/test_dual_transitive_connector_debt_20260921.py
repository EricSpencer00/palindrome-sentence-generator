import dual_transitive_connector_debt_20260921 as experiment


def test_dual_transitive_arms_have_real_connector_and_exact_gate():
    result = experiment.run()
    assert result["stats"]["controls"] == 9
    assert result["stats"]["max_matched_prefix"] == 1
    assert result["stats"]["accepted_exact"] == 0
    assert result["novelty_preflight"]["status"] == "passed"
    assert any(row["gates"]["content_disjoint"] for row in result["rendered_controls"])
