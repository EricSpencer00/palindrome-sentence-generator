from experiments.variable_phrase_grammar_20260920 import audit, compatible, consume_residual, grammar_paths, run


def test_overlap_invariant_is_character_level():
    assert consume_residual("anaide", "anaid") == ("e", "")
    assert consume_residual("anaide", "ariver") is None
    assert compatible("an aide", "ediana")


def test_paths_are_variable_and_not_fixed_six_slots():
    paths = grammar_paths()
    assert len({len(p) for p in paths}) >= 3
    assert all(p[:2] == ("NP", "VP") for p in paths)


def test_run_is_independently_audited():
    result = run(limit=3000)
    assert result["bank_sizes"]["NP"] >= 7
    for row in result["exact_candidates"]:
        assert audit(row["rendered"]) == row["audit"]
