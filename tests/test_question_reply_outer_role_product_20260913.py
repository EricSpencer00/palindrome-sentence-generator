"""Tests for the productive question/reply outer-role topology."""

from experiments.question_reply_outer_role_product_20260913 import (
    independent_discourse_parse,
    productive_derivations,
    replay_character_ledger,
    run,
)


def test_outer_role_inventory_is_productive_and_coherent():
    derivations = productive_derivations()
    result = run(state_limit=100_000)

    assert result["config"]["productive_typed_role_alternatives"]
    assert result["config"]["short_functional_question_operator"] == "are"
    assert result["config"]["semantically_normal_final_reply_object"]
    assert result["derivation_count"] == len(derivations) == 21
    assert all(independent_discourse_parse(d, d.rendered)["ok"] for d in derivations)


def test_question_and_reply_streams_are_reversed_before_pairing():
    result = run(state_limit=100_000)

    for left, right in zip(result["left_grammar_rows"], result["right_grammar_rows"]):
        assert right["reversed_suffix_tape"] == right["forward_suffix_tape"][::-1]
        assert left["relation"] == right["relation"]
    assert result["config"]["complete_pair_selected_after_intersection"]
    assert result["config"]["source_reverse_terminal_compatibility_constraint"]


def test_frontier_and_exhaustion_are_persisted_without_fake_candidate():
    result = run(state_limit=100_000)
    failure = result["first_jointly_reachable_failure"]

    assert result["states_exhausted"] is True
    assert result["intersection"]["search_truncated"] is False
    assert result["intersection"]["states_examined"] == 5
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


def test_terminal_ledger_replay_is_fail_closed_and_bounded_runs_report_truncation():
    result = run(state_limit=1)
    assert result["states_exhausted"] is False
    assert result["intersection"]["search_truncated"] is True
    synthetic = {
        "question_tape": "arep",
        "reversed_reply_tape": "arep",
        "character_ledger": [
            {"position": 1, "left_character": "a", "right_reversed_character": "a", "matched": True},
            {"position": 2, "left_character": "r", "right_reversed_character": "r", "matched": True},
            {"position": 3, "left_character": "e", "right_reversed_character": "e", "matched": True},
            {"position": 4, "left_character": "p", "right_reversed_character": "p", "matched": True},
        ],
    }
    assert replay_character_ledger(synthetic)

