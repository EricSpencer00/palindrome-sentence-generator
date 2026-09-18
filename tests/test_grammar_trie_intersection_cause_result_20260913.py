"""Tests for the cause/result grammar-product lexical bridge."""

from experiments.grammar_trie_intersection_cause_result_20260913 import (
    compile_grammars,
    intersect_tries,
    run,
)


def test_clause_grammars_compile_before_pair_materialization():
    result = run(state_limit=100_000)

    assert result["config"]["grammar_product_intersection"] is True
    assert result["config"]["complete_pair_selected_after_intersection"] is True
    assert [row["rendered_clause"] for row in result["left_grammar_rows"]] == [
        "A skilled baker warms fresh bread.",
        "A careful farmer carries clean water.",
    ]
    assert [row["rendered_clause"] for row in result["right_grammar_rows"]] == [
        "The scent fills the area.",
        "The stream reaches the field.",
    ]


def test_right_grammar_is_intersected_as_reversed_character_stream():
    left_root, right_root, left_rows, right_rows = compile_grammars()
    pairs, info = intersect_tries(left_root, right_root, state_limit=100_000)

    assert pairs == []
    assert info["states_examined"] == 2
    assert info["deepest_joint_state"]["prefix"] == "a"
    assert all(row["reversed_tape"] == row["forward_tape"][::-1] for row in right_rows)
    assert all(row["tape"] for row in left_rows)


def test_zero_survivor_persists_first_joint_literal_failure_and_no_fake_candidate():
    result = run(state_limit=100_000)
    failure = result["first_jointly_reachable_failure"]

    assert failure == {
        "prefix": "a",
        "left_next_chars": ["c", "s"],
        "right_next_chars": ["e"],
        "reason": "no_common_next_character",
    }
    assert result["records"] == []
    assert result["exact_candidates"] == []
    assert result["parsed_exact_candidates"] == []
    assert result["mechanically_admitted"] == []
    assert result["intersection"]["terminal_pairs_rejected_during_expansion"] == []

