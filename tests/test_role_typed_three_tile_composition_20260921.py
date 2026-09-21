import role_typed_three_tile_composition_20260921 as experiment


def test_role_typed_three_tile_composition_is_a_nonexact_control_lane():
    result = experiment.run()
    assert result["stats"]["paths"] == 6
    assert result["stats"]["role_compatible"] == 6
    assert result["stats"]["accepted_exact"] == 0
    assert result["novelty_preflight"]["status"] == "passed"
    assert all(row["gates"]["complete_grammatical_template"] for row in result["rendered_controls"])
    assert all(row["gates"]["content_disjoint"] for row in result["rendered_controls"])
    assert all(not row["accepted"] for row in result["rendered_controls"])
    assert all(row["provenance"]["human_readability_claim"].startswith("grammatical template") for row in result["rendered_controls"])

