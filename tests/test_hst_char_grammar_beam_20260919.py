from experiments.hst_char_grammar_beam_20260919 import audit, search

def test_independent_audit():
    a = audit("Able was I ere I saw Elba")
    assert a["two_pointer_exact"] and a["sha_equal"]

def test_character_beam_is_bounded_and_no_repair():
    r = search(38, beam=20, max_nodes=200)
    assert r["nodes"] <= 200
    assert r["residual_reason"] == "live character obligation or bounded beam exhausted" or r["exact"]
