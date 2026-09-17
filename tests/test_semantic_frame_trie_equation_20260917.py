from experiments.semantic_frame_trie_equation_20260917 import audit, run, search, normalize


def test_independent_frames_and_joint_boundary_witnesses():
    result = run()
    assert result["experiment_id"] == "semantic-frame-trie-equation-20260917"
    assert result["provenance"]["catalogue_read"] is False
    assert result["provenance"]["fixed_tape"] is False
    assert result["frontiers"]
    row = result["frontiers"][0]
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
