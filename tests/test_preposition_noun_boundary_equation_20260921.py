from preposition_noun_boundary_equation_20260921 import audit, banks, consume, run

def test_finite_boundary_equation_contract():
    assert len(banks()) == 6
    assert consume("abc", "cba") == ("", "")
    assert consume("abc", "xyz") is None
    result = run(state_limit=5000)
    assert result["stats"]["states"] > 0
    assert result["provenance"]["search_from_equation"]
    for row in result["exact_candidates"]:
        assert row["audit"] == audit(row["rendered"])
        assert row["provenance"]["finished_tape_reversal"] is False
