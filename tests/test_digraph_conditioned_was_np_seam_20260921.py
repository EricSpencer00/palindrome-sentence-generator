import digraph_conditioned_was_np_seam_20260921 as experiment


def test_digraph_conditioned_seam_retains_natural_controls():
    result = experiment.run()
    assert result["stats"]["pairs"] == 8
    assert result["stats"]["matched_two_plus"] == 8
    assert result["stats"]["max_matched_prefix"] == 7
    assert result["stats"]["accepted_exact"] == 0
    assert result["novelty_preflight"]["status"] == "passed"
    assert all(row["gates"]["roles_coherent"] for row in result["rendered_controls"])
    assert all(not row["accepted"] for row in result["rendered_controls"])
