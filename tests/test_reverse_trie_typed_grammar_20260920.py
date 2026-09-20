from experiments.reverse_trie_typed_grammar_20260920 import audit, run, search, _trie

def test_reverse_trie_and_forward_banks_are_distinct():
    trie = _trie([("the", "calm")])
    node = trie
    for ch in "mlac eht".replace(" ", ""):
        assert ch in node
        node = node[ch]
    assert "$" in node

def test_lane_records_live_unequal_residuals_and_independent_audit():
    result = run()
    assert result["status"] == "frontier_exhausted_no_exact"
    assert result["stats"]["exact"] == 0
    assert result["stats"]["reader_eligible_exact"] == 0
    row = (result["candidates"] or result["diagnostics"])[0]
    assert row["residual_left"] or row["residual_right_reverse_facing"]
    assert row["audit"] == audit(row["rendered"])
    assert row["provenance"]["reverse_trie_walk_before_render"]
    assert row["provenance"]["finished_tape_reversal"] is False
    assert row["provenance"]["repeated_units"] is False
    assert row["provenance"]["malformed_surface"] is False
    assert row["provenance"]["reader_eligible"] is False
    assert row["quarantine"]["reader_eligible"] is False
    assert row["audit"]["forward_sha256"] != row["audit"]["reverse_sha256"]
