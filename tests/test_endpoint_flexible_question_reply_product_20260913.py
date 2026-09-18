"""Tests for endpoint-flexible typed question/reply generation."""

from experiments.endpoint_flexible_question_reply_product_20260913 import (
    endpoint_compatibility_automaton,
    independent_discourse_parse,
    productive_derivations,
    replay_character_ledger,
    run,
)


def test_endpoint_automaton_selects_multiple_semantic_operator_object_pairs():
    endpoints = endpoint_compatibility_automaton()
    result = run(state_limit=100_000)

    assert result["config"]["endpoint_automaton_before_full_derivation"]
    assert result["config"]["operator_final_object_selected_jointly"]
    assert len(result["endpoint_automaton_inventory"]) == 16
    assert {pair.operator for pair in endpoints} == {"can", "are", "do"}
    assert "will" in result["config"]["multiple_ordinary_question_operators"]
    assert all(len(pair.shared_prefix) >= len(pair.operator) + 1 for pair in endpoints)
    assert {pair.final_object for pair in endpoints} == {"almanac", "opera", "method"}
    assert all(not row["crosses_operator_subject_boundary"] for row in result["endpoint_automaton_inventory"] if row["operator"] == "will")
    assert result["derivation_count"] == len(productive_derivations()) == 24
    assert all(independent_discourse_parse(derivation, derivation.rendered)["ok"] for derivation in productive_derivations())


def test_compatibility_pairs_are_compiled_into_reversed_tries():
    result = run(state_limit=100_000)

    assert len(result["endpoint_compatibility_pairs"]) == 6
    for left, right in zip(result["left_grammar_rows"], result["right_grammar_rows"]):
        assert right["reversed_suffix_tape"] == right["forward_suffix_tape"][::-1]
        assert left["relation"] == right["relation"]
    assert result["config"]["complete_pair_selected_after_intersection"]
    assert result["config"]["match_crosses_operator_subject_boundary"]


def test_search_persists_cross_boundary_frontier_without_claiming_candidate():
    result = run(state_limit=100_000)
    failure = result["first_jointly_reachable_failure"]

    assert result["states_exhausted"] is True
    assert result["intersection"]["search_truncated"] is False
    assert result["intersection"]["states_examined"] == 12
    assert result["intersection"]["deepest_joint_state"]["prefix"] == "arep"
    assert failure == {
        "prefix": "arep",
        "left_next_chars": ["a", "e"],
        "right_next_chars": ["o"],
        "reason": "no_common_next_character",
    }
    assert result["records"] == []
    assert result["exact_candidates"] == []
    assert result["parsed_exact_candidates"] == []
    assert result["mechanically_admitted"] == []


def test_ledger_is_replayable_and_truncation_is_explicit():
    result = run(state_limit=1)
    assert result["states_exhausted"] is False
    assert result["intersection"]["search_truncated"] is True
    synthetic = {
        "question_tape": "cana",
        "reversed_reply_tape": "cana",
        "character_ledger": [
            {"position": 1, "left_character": "c", "right_reversed_character": "c", "matched": True},
            {"position": 2, "left_character": "a", "right_reversed_character": "a", "matched": True},
            {"position": 3, "left_character": "n", "right_reversed_character": "n", "matched": True},
            {"position": 4, "left_character": "a", "right_reversed_character": "a", "matched": True},
        ],
    }
    assert replay_character_ledger(synthetic)
