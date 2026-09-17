from experiments.semantic_frame_trie_equation_20260917 import audit, run, search, normalize


def test_quarantined_fixed_frame_diagnostic_has_truthful_provenance():
    result = run()
    assert result["experiment_id"] == "semantic-frame-trie-equation-20260917"
    assert result["provenance"]["catalogue_read"] is False
    assert result["provenance"]["fixed_tape"] is True
    assert result["status"] == "quarantined_no_search_diagnostic"
    assert result["provenance"]["generative_search"] is False
    assert result["provenance"]["trie_constrained_transitions"] == 0
    assert result["provenance"]["boundary_choices_searched"] == 0
    assert result["provenance"]["fixed_frame_pairs_compared"] == 16
    assert result["frontiers"]
    row = result["frontiers"][0]
    assert row["boundary_choices"]["left_words"][:2] == ["a", "courier"]
    assert row["trie_nodes_left"] > len(row["boundary_choices"]["left_words"])
    assert row["trie_nodes_right"] > len(row["boundary_choices"]["right_words"])
    assert row["audit"] == audit(row)


def test_audit_catches_non_palindrome_and_recomputes_independently():
    result = search()
    row = result["frontiers"][0]
    assert row["exact"] is False
    assert row["normalized_left"] == normalize(row["left"])
    assert row["normalized_right"] == normalize(row["right"])
    assert audit(row)["left_equals_reverse_right"] is False
