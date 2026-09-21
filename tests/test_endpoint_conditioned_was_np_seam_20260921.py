import endpoint_conditioned_was_np_seam_20260921 as experiment


def test_endpoint_conditioned_seam_counts_only_successful_matches():
    result = experiment.run()
    assert result["stats"]["pairs"] == 8
    assert result["stats"]["seam_advanced"] == 8
    assert result["stats"]["max_matched_prefix"] == 9
    assert result["stats"]["online_closed"] == 0
    assert result["stats"]["accepted_exact"] == 0
    assert result["novelty_preflight"]["status"] == "passed"
    assert all(row["matched_prefix"] > 3 for row in result["rendered_controls"])
    assert all(not row["accepted"] for row in result["rendered_controls"])
