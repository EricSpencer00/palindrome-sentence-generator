from incremental_character_grammar_product_20260921 import audit, run

def test_incremental_product_has_online_controls_and_trace():
    out = run()
    assert out["novelty_preflight"]["status"] == "passed"
    assert out["rendered_controls"]
    assert out["stats"]["transitions"] <= 60
    assert all(r["provenance"]["selected_online"] for r in out["rendered_controls"])
    assert all("bilateral_obligation_trace" in r for r in out["rendered_controls"])

def test_audit_independently_hashes_forward_and_reverse():
    a = audit("A man, a plan, a canal: Panama")
    assert a["pointer_exact"]
    assert a["sha256_forward"] == a["sha256_reverse"]
