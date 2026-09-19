from experiments.cfg_character_product_20260918 import audit, run

def test_audit_independently_checks_both_directions():
    result = audit("An aide rips nine memos; some men inspire Diana.")
    assert result["two_pointer_exact"]
    assert result["sha256_forward"] == result["sha256_reverse"]

def test_cfg_product_is_bounded_and_keeps_controls_when_empty():
    result = run()
    assert result["derivations"] == 1500
    assert result["exact_count"] == 0
    assert result["candidate_count"] >= 2
    assert all(row["reader_gate"] == "closed" for row in result["candidates"])
