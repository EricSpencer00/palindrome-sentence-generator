import typed_edge_tile_composition_20260921 as experiment


def test_typed_edge_composition_filters_before_rendering():
    result = experiment.run()
    assert result["stats"]["controls"] == 4
    assert result["stats"]["typed_compatible"] == 2
    assert result["stats"]["accepted_exact"] == 0
    assert result["novelty_preflight"]["status"] == "passed"
    compatible = [row for row in result["rendered_controls"] if row["gates"]["typed_edges_compatible"]]
    assert all(row["gates"]["content_disjoint"] for row in compatible)
    assert all(not row["accepted"] for row in result["rendered_controls"])

