import are_np_adjective_in_an_era_20260921 as experiment


def test_grammatical_era_shell_is_exact_gated():
    result = experiment.run()
    assert result["stats"]["controls"] == 9
    assert result["stats"]["shell_match_two_plus"] == 9
    assert result["stats"]["accepted_exact"] == 0
    assert result["novelty_preflight"]["status"] == "passed"
    assert all(not row["accepted"] for row in result["rendered_controls"])
