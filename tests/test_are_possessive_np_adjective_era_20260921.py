import are_possessive_np_adjective_era_20260921 as experiment


def test_possessive_era_shell_uses_complete_noun_phrases():
    result = experiment.run()
    assert result["stats"]["controls"] == 9
    assert result["stats"]["shell_match_two_plus"] == 9
    assert result["stats"]["accepted_exact"] == 0
    assert result["novelty_preflight"]["status"] == "passed"
    assert all("'" in row["np"] and row["gates"]["shell_grammatical"] for row in result["rendered_controls"])
