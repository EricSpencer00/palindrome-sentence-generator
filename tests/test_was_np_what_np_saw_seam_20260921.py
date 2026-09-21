import was_np_what_np_saw_seam_20260921 as experiment


def test_was_what_saw_seam_reports_no_false_closure():
    result = experiment.run()
    assert result["stats"]["pairs"] == 16
    assert result["stats"]["online_closed"] == 0
    assert result["stats"]["boundary_extended"] == 0
    assert result["stats"]["accepted_exact"] == 0
    assert result["novelty_preflight"]["status"] == "passed"
    assert all(not row["accepted"] for row in result["rendered_controls"])
