from experiments.reverse_segmentation_clause_index_20260919 import _audit, run


def test_audit_independently_verifies_known_seed():
    result = _audit("An aide rips nine memos; some men inspire Diana.")
    assert result["two_pointer"] is True
    assert result["sha_match"] is True
    assert result["mechanical"] is True
    assert result["length"] == 38


def test_indexed_search_has_reproducible_provenance_and_repair():
    result = run()
    assert result["experiment"] == "reverse-segmentation-clause-index-20260919"
    assert result["clause_count"] > 1000
    assert "partial reverse prefixes" in result["next_repair"]
    for row in result["candidates"]:
        assert row["provenance"] == result["experiment"]
        assert row["audit"]["two_pointer"]
        assert row["audit"]["sha_match"]
