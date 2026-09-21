import are_plural_np_adjective_era_20260921 as experiment


def test_plural_era_shell_keeps_agreement_and_exact_gates():
    result = experiment.run()
    assert result["stats"]["controls"] == 9
    assert result["stats"]["shell_match_two_plus"] == 9
    assert result["stats"]["accepted_exact"] == 0
    assert result["novelty_preflight"]["status"] == "passed"
    assert all(row["gates"]["plural_agreement"] for row in result["rendered_controls"])
