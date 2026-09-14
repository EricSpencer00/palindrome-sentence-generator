import pytest

from llm_palindrome.frontier_macros import enumerate_macros, state_from_anchors, witness_preserves_anchors
from llm_palindrome.search import WordTries


def test_anchors_track_unmatched_letters():
    state = state_from_anchors("Some", "memos")
    assert (state.overhang, state.side) == ("m", "R")
    with pytest.raises(ValueError, match="incompatible"):
        state_from_anchors("Some", "letters")


def test_completion_preserves_both_anchors_without_overlap():
    state = state_from_anchors("Some", "memos")
    assert witness_preserves_anchors(state, "Some managers revised the memos.")
    assert not witness_preserves_anchors(state, "Several managers revised the memos.")
    assert not witness_preserves_anchors(state, "Some managers revised the letters.")
    assert not witness_preserves_anchors(state, "Some memos.")


def test_macros_replay_exact_debt_and_allow_cross_word_boundaries():
    state = state_from_anchors("Some", "memos")
    tries = WordTries(["more", "er", "re", "met", "te", "a", "ma", "am", "me", "em", "memo", "omen"])
    moves = enumerate_macros(state, tries, min_words=2, max_words=4, menu_size=30)
    assert moves
    for move in moves:
        left = "".join(move.state.left)
        right = "".join(move.state.right)[::-1]
        if move.state.side == "L":
            assert left == right + move.state.overhang
        else:
            assert right == left + move.state.overhang
        assert 2 <= move.menu_item()["added_words"] <= 4
        assert len(move.move_id) == 16
