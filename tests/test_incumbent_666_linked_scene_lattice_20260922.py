import json

from experiments.incumbent_666_linked_scene_lattice_20260922 import OUT


def test_bidirectional_trie_finds_exact_novel_clause_pair_with_live_obligations():
    payload = json.loads(OUT.read_text())
    row = payload["rows"][0]
    lattice = row["typed_trie_intersection"]
    result = lattice["result"]
    selected = lattice["selected_candidate"]
    assert lattice["predicate_inventory"] == ["saw", "spots", "stops", "was"]
    assert lattice["targeted_subjects"] == ["Aidan", "Aram", "Aras", "Aron", "Mara", "Nadia", "Sara"]
    assert row["alternate_seam"]["normalized_windows"] == {"left": [127, 140], "right": [526, 539]}
    assert row["alternate_seam"]["raw_windows"] == {"left": [171, 188], "right": [719, 736]}
    assert result["expansions"] <= lattice["max_paired_expansions"] == 8
    assert result["closures"]
    assert selected["left"] == result["closures"][0]["left_clause"]
    assert selected["right"] == result["closures"][0]["right_clause"]
    assert selected["normalized_letters_per_side"] == 13
    assert selected["both_active_entities"] == {
        "left_before": "Aras",
        "left_after": "Aram",
        "right_before": "Aidan",
        "right_after": "Sara",
    }
    assert result["states"][0]["owner"] == "grammar-trie-intersection"
    assert result["states"][0]["left_cursor"] == 1
    assert result["states"][0]["right_reverse_cursor"] == 1
    assert result["states"][0]["left_residual"] == 12
    assert result["states"][0]["right_reverse_residual"] == 12
    assert all({"left_cursor", "right_reverse_cursor", "left_residual", "right_reverse_residual", "active_entities"} <= state.keys() for state in result["states"])
    assert result["rejected"][0]["parent_frame_reuse"] == ["mara|stops"]


def test_exact_child_is_independently_verified_and_shell_spaces_are_preserved():
    payload = json.loads(OUT.read_text())
    row = payload["rows"][0]
    lattice = row["typed_trie_intersection"]
    selected = lattice["selected_candidate"]
    audit = row["independent_audit"]
    spacing = selected["spacing"]
    assert row["promotion_status"]["promoted"] is False
    assert lattice["admission"] == {
        "accepted": True,
        "exact_child_saved": True,
        "independently_exact": True,
        "new_event_content": True,
    }
    assert audit["normalized_letters"] == 666
    assert audit["two_pointer_exact"] is True
    assert len(selected["sha256"]) == 64
    assert spacing["left_leading_preserved"] is True
    assert spacing["right_leading_preserved"] is True
    assert spacing["left_trailing_preserved"] is True
    assert spacing["right_trailing_preserved"] is True
    assert spacing["spacing_shell_preserved"] is True
    assert lattice["selected_closure"]["parent_frame_reuse"] == []
    assert lattice["selected_closure"]["parent_clause_reuse"] == []
    assert lattice["selected_closure"]["incremental_novelty"] == {
        "parent_frames_clear": True,
        "parent_clauses_clear": True,
        "used_frames_clear": True,
        "used_clauses_clear": True,
    }
