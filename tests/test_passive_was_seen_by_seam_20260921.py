import passive_was_seen_by_seam_20260921 as experiment


def test_passive_shell_negative_result_is_explicit():
    result = experiment.run()
    assert result["stats"]["pairs"] == 9
    assert result["stats"]["usable_boundaries"] == 0
    assert result["stats"]["accepted_exact"] == 0
    assert result["novelty_preflight"]["status"] == "passed"
    assert all(not row["accepted"] for row in result["rendered_controls"])
