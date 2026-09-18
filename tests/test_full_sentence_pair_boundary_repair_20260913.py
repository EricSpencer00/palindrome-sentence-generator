from experiments.full_sentence_pair_boundary_repair_20260913 import debt, independent_audit, search


def test_character_debt_ignores_word_boundaries():
    assert debt("a red", "der a")["compatible"]
    assert not debt("a red", "blue a")["compatible"]


def test_bounded_run_is_explicitly_invalidated_and_retains_diagnostics():
    result = search(beam=10, limit=80)
    assert result["status"] == "invalidated_missing_connected_feature_grammar"
    assert result["stats"]["complete_grammar_rejections"] > 0
    assert not result["admitted"]
    assert result["candidate_use"].startswith("forbidden")
    for row in result["rejections"]:
        assert row["independent_exact"]["exact"] is False
        assert "semantic_witness" in row
