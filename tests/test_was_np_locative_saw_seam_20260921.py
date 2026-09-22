import experiments.was_np_locative_saw_seam_20260921 as experiment


def test_locative_np2_seam_reports_reader_controls_without_false_closure():
    result = experiment.run()
    assert result["stats"]["pairs"] == 16
    assert result["stats"]["online_closed"] == 0
    assert result["stats"]["boundary_extended"] == 0
    assert result["stats"]["accepted_exact"] == 0
    assert result["novelty_preflight"]["status"] == "passed"
    assert any(row["gates"]["no_repeated_units"] for row in result["rendered_controls"])
