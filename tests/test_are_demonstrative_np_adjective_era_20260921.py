import experiments.are_demonstrative_np_adjective_era_20260921 as experiment


def test_demonstrative_era_shell_uses_agreeing_plural_nps():
    result = experiment.run()
    assert result["stats"]["controls"] == 9
    assert result["stats"]["shell_match_two_plus"] == 9
    assert result["stats"]["accepted_exact"] == 0
    assert result["novelty_preflight"]["status"] == "passed"
    assert all(row["gates"]["demonstrative_agreement"] for row in result["rendered_controls"])
