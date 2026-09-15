from experiments.dual_parse_bridge_20260914 import audit, run, sentences

def test_audit_independently_accepts_exact_tape():
    assert audit("drawer reward")["exact"] is True
    assert audit("drawer reward!")["normalized_tape"] == "drawerreward"

def test_sentence_product_has_live_typed_chunks():
    rows = sentences(100)
    assert rows
    assert all(row["chunks"] and row["chunks"][0]["kind"] == "NP" for row in rows)
    assert all(any(c["kind"] == "VP" for c in row["chunks"]) for row in rows)
    import re
    assert all(len((words := re.findall(r"[a-z]+", row["text"].lower()))) == len(set(words)) for row in rows)

def test_bounded_dual_parse_search_is_reproducible_and_does_not_claim_readability():
    result = run(500)
    assert result["config"]["staggered_chunk_boundaries"] is True
    assert result["status"] == "dual_parse_exact_closures_need_blinded_readability"
    assert result["candidate_count"] == 0
