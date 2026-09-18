"""Tests for the productive shared-event action/result search."""

from experiments.coupled_event_so_grammar_product_20260913 import (
    independent_discourse_parse,
    productive_derivations,
    replay_character_ledger,
    run,
)


def test_inventory_is_productive_and_agreement_safe():
    derivations = productive_derivations()
    result = run(state_limit=100_000)

    assert result["config"]["productive_typed_role_alternatives"]
    assert result["config"]["shared_event_relation_in_both_clauses"]
    assert result["config"]["finite_inventory_exhausted_not_global_search_complete"]
    assert "no claim of global grammar-search completeness" in result["completeness_statement"]
    assert result["derivation_count"] == len(derivations) == 224
    assert len(result["left_grammar_rows"]) == len(result["right_grammar_rows"]) == 224
    assert {row["relation"] for row in result["left_grammar_rows"]} == {
        "baking_aroma", "planting_growth", "painting_display"
    }
    assert all(independent_discourse_parse(d, d.rendered)["ok"] for d in derivations)


def test_reversed_result_suffix_is_compiled_before_terminal_pairing():
    result = run(state_limit=100_000)

    for left, right in zip(result["left_grammar_rows"], result["right_grammar_rows"]):
        assert right["reversed_suffix_tape"] == right["forward_suffix_tape"][::-1]
        assert left["relation"] == right["relation"]
    assert result["config"]["complete_pair_selected_after_intersection"]
    assert result["config"]["source_reverse_terminal_compatibility_constraint"]


def test_productive_search_records_real_frontier_without_fake_closure():
    result = run(state_limit=100_000)
    failure = result["first_jointly_reachable_failure"]

    assert result["intersection"]["states_examined"] == 3
    assert result["intersection"]["deepest_joint_state"]["prefix"] == "an"
    assert failure == {
        "prefix": "an",
        "left_next_chars": ["a"],
        "right_next_chars": ["e"],
        "reason": "no_common_next_character",
    }
    assert result["records"] == []
    assert result["exact_candidates"] == []
    assert result["parsed_exact_candidates"] == []
    assert result["mechanically_admitted"] == []


def test_bounded_status_is_explicit_and_terminal_ledger_is_replayable():
    complete = run(state_limit=100_000)
    truncated = run(state_limit=1)

    assert complete["states_exhausted"] is True
    assert complete["intersection"]["search_truncated"] is False
    assert truncated["states_exhausted"] is False
    assert truncated["intersection"]["search_truncated"] is True
    synthetic = {
        "action_tape": "abc",
        "reversed_suffix_tape": "abc",
        "character_ledger": [
            {"position": 1, "left_character": "a", "right_reversed_character": "a", "matched": True},
            {"position": 2, "left_character": "b", "right_reversed_character": "b", "matched": True},
            {"position": 3, "left_character": "c", "right_reversed_character": "c", "matched": True},
        ],
    }
    assert replay_character_ledger(synthetic)
