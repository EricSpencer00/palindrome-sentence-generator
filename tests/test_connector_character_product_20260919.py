from experiments.connector_character_product_20260919 import audit, search, consume

def test_independent_audit_accepts_known_seed():
    a = audit("An aide rips nine memos; some men inspire Diana.")
    assert a["two_pointer_exact"] and a["sha_equal"]

def test_product_consumes_live_edges_without_reverse_claim():
    assert consume("ab", "ba")[0]
    assert not consume("ab", "ca")[0]

def test_search_is_bounded_and_keeps_provenance():
    r = search(2)
    assert r["stats"]["rendered"] > 0
    assert r["provenance"]["search_uses_finished_reversal"] is False
    assert all("reader_status" in x for x in r["actual_candidates"])
    assert any(x["reader_status"].startswith("frontier") for x in r["actual_candidates"])
