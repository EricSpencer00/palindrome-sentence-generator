import experiments.constructive_qa_character_debt_20260921 as experiment


def test_constructive_qa_debt_is_online_and_has_no_false_exact_candidate():
    result = experiment.run()
    assert result["stats"]["pairs"] == 4096
    assert result["stats"]["online_closed"] == 0
    assert result["stats"]["accepted_exact"] == 0
    assert result["novelty_preflight"]["status"] == "passed"
    assert all(not row["accepted"] for row in result["rendered_controls"])
    assert all(row["provenance"]["selected_online_against_opposing_character_debt"] for row in result["rendered_controls"])
