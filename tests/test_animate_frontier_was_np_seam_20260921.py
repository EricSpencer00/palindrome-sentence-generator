import animate_frontier_was_np_seam_20260921 as experiment


def test_animate_frontier_does_not_claim_unobserved_extension():
    result = experiment.run()
    assert result["stats"]["pairs"] == 16
    assert result["stats"]["advanced_beyond_12"] == 0
    assert result["stats"]["accepted_exact"] == 0
    assert result["frontier_control"]["matched_prefix_length"] == 12
    assert result["frontier_control"]["residual"] == ["a", "t"]
    assert result["novelty_preflight"]["status"] == "passed"
