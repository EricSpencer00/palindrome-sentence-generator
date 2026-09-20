from experiments.earley_scene_chart_orbit_20260920 import audit, complete_derivations, consume, run


def test_recursive_chart_has_complete_ordinary_controls():
    paths = complete_derivations()
    assert paths
    rendered = [" ".join(x.text for x in path) for path in paths]
    assert any("in the hall" in text for text in rendered)
    assert any("while" in text or "and" in text for text in rendered)


def test_character_chart_and_independent_audit():
    assert consume("abc", "cba") == ("", "")
    assert consume("abc", "ba") == ("c", "")
    assert consume("abc", "xyz") is None
    result = run(state_limit=20_000)
    assert result["stats"]["grammar_paths"] > 0
    assert result["provenance"]["variable_word_boundaries"]
    for row in result["candidates"]:
        assert row["audit"] == audit(row["rendered"])
        assert row["audit"]["exact"]
