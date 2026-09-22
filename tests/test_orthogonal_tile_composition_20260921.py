import experiments.orthogonal_tile_composition_20260921 as experiment


def test_tile_composition_records_boundary_failure_without_shortcuts():
    result = experiment.run()
    assert result["stats"]["controls"] == 12
    assert result["stats"]["max_boundary_obligation"] == 0
    assert result["stats"]["accepted_exact"] == 0
    assert result["novelty_preflight"]["status"] == "passed"
    assert any(row["gates"]["content_disjoint"] for row in result["rendered_controls"])
    assert all(not row["accepted"] for row in result["rendered_controls"])

