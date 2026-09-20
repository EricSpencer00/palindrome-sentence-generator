from experiments.typed_two_event_relative_grammar_20260920 import audit, consume, paths, run


def test_residual_consumption_is_one_sided_and_exact():
    assert consume("abc", "ab") == ("c", "")
    assert consume("abc", "ax") is None


def test_variable_typed_paths():
    ps = paths()
    assert len(ps) > 1
    assert len({len(p) for p in ps}) > 1
    assert all(p[:4] == ("NP", "VP", "NP", "VP") for p in ps)


def test_run_rows_have_independent_audit():
    result = run(limit=12000)
    for row in result["exact_candidates"]:
        assert row["audit"] == audit(row["rendered"])
        assert row["audit"]["letters"] > 38
