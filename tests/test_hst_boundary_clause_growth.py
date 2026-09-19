from experiments.hst_boundary_clause_growth_20260919 import audit, search


def test_audit_independently_accepts_exact_text():
    result = audit("Able was I ere I saw Elba")
    assert result["two_pointer_exact"]
    assert result["sha_equal"]


def test_boundary_search_is_bounded_and_never_repairs():
    result = search(40, max_nodes=2_000)
    assert result["nodes"] <= 2_000
    for row in result["candidates"]:
        assert row["provenance"]["indexed_before_render"]
        assert not row["provenance"]["post_hoc_repair"]
        assert row["audit"]["letters"] == 40
