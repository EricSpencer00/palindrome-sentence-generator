import experiments.copular_question_seam_20260921 as experiment


def test_copular_terminal_pairs_do_not_certify_fragments():
    result = experiment.run()
    assert result["stats"]["controls"] == 27
    assert result["stats"]["valid_terminal_pairs"] == 3
    assert result["stats"]["natural_shell_controls"] == 0
    assert result["stats"]["accepted_exact"] == 0
    assert result["novelty_preflight"]["status"] == "passed"
    assert all(not row["accepted"] for row in result["rendered_controls"])
