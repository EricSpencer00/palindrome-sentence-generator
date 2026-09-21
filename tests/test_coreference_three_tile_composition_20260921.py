import coreference_three_tile_composition_20260921 as experiment


def test_coreference_control_keeps_lexical_units_distinct():
    result = experiment.run()
    assert result["stats"]["paths"] == 1
    assert result["stats"]["coreference_compatible"] == 1
    assert result["stats"]["accepted_exact"] == 0
    row = result["rendered_controls"][0]
    assert row["gates"]["coreference_state_valid"]
    assert row["gates"]["lexical_content_disjoint"]
    assert row["provenance"]["readability_claim"].startswith("grammatical control")
    assert not row["accepted"]

