from experiments.two_clause_irreducible_dual_parse_20260921 import run


def test_two_clause_search_is_bounded_and_irreducible():
    result = run()
    assert result["stats"]["cap_reached"] is False
    assert result["stats"]["states"] > 0
    assert result["stats"]["intermediate_closure_rejections"] >= 0
    assert result["reader_packet"] == []


def test_every_two_clause_closure_uses_central_admission():
    result = run()
    for row in result["exact_candidates"]:
        assert row["audit"]["two_pointer_exact"] is True
        assert row["semantic_topology"] == ["A", "B", "B-prime", "A-prime"]
        assert row["mechanically_admitted"] == all(row["mechanical_admission"].values())
