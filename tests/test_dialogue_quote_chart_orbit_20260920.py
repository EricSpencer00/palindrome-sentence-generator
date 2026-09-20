from experiments.dialogue_quote_chart_orbit_20260920 import audit, complete_derivations, consume, run


def test_dialogue_paths_contain_complete_typed_quotes():
    paths = complete_derivations()
    assert paths
    assert any(any(x.symbol == "QUOTE" for x in path) for path in paths) or any(any(x.quote_depth for x in path) for path in paths)
    assert any("that" in " ".join(x.text for x in path) for path in paths)


def test_live_chart_and_independent_audit():
    assert consume("abc", "cba") == ("", "")
    assert consume("abc", "ba") == ("c", "")
    assert consume("abc", "xyz") is None
    result = run(state_limit=20_000)
    assert result["stats"]["grammar_paths"] > 0
    for row in result["candidates"]:
        assert row["audit"] == audit(row["rendered"])
        assert row["audit"]["exact"]
