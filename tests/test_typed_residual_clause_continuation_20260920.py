from experiments.typed_residual_clause_continuation_20260920 import consume, run


def test_consume_retains_debt_between_chunks():
    debt, checked, mismatch = consume("ab", "a")
    assert (debt, checked, mismatch) == ("b", 1, None)
    debt, checked, mismatch = consume("c", "cb", debt)
    assert (debt, checked, mismatch) == ("", 2, None)


def test_run_records_keyed_pre_render_decisions_and_controls():
    result = run()
    assert result["stats"]["continuation_keys"] == 4
    assert result["stats"]["continuations"] == 8
    assert result["stats"]["exact_above_38"] == 0
    assert all(row["provenance"]["key_selected_before_render"] for row in result["diagnostic_candidates"])
