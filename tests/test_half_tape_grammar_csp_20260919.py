from experiments.half_tape_grammar_csp_20260919 import _audit, _place, run


def test_half_tape_alias_rejects_conflicting_character_assignments():
    # In a four-letter target positions 0 and 3 alias the same variable.
    assert _place([None, None], 0, ("ab",), 4) is not None
    assert _place(["a", None], 3, ("b",), 4) is None


def test_half_tape_pilot_records_exact_rows_with_independent_audits():
    result = run(lengths=range(38, 41), max_nodes=30_000)
    assert result["stats"]["exact"] >= 1
    assert result["stats"]["longest_exact"] >= 38
    assert result["exact_candidates"]
    for row in result["exact_candidates"]:
        assert row["audit"]["two_pointer_exact"]
        assert row["audit"]["sha_equal"]
        assert row["audit"]["normalized"] == row["audit"]["normalized"][::-1]
