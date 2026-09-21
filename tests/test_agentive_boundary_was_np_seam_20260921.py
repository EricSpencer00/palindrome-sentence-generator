import agentive_boundary_was_np_seam_20260921 as experiment


def test_agentive_object_by_boundary_is_grammatical_and_exact_gated():
    result = experiment.run()
    assert result["stats"]["pairs"] == 8
    assert result["stats"]["max_matched_prefix"] == 12
    assert result["stats"]["accepted_exact"] == 0
    assert result["novelty_preflight"]["status"] == "passed"
    assert all(row["gates"]["coherent_roles"] for row in result["rendered_controls"])
    assert all(not row["accepted"] for row in result["rendered_controls"])
