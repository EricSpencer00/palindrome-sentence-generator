from experiments.shared_object_event_product_20260930 import independent, run


def test_controls_are_ordinary_and_not_exact_shortcuts():
    result = run()
    assert result["controls"]
    assert all(not row["audit"]["exact"] for row in result["controls"])
    assert result["provenance"]["complete_sentence_enumeration"] is False


def test_independent_audit_rejects_nonpalindrome():
    row = independent("The nurse reads the note, and she files it.")
    assert not row["exact"]
    assert not row["validator_exact"]
    assert row["sha_equal"] is False


def test_search_is_live_product():
    result = run()
    assert result["search"]["states"] > 0
    assert result["search"]["grammar_character_edges"] > 0
