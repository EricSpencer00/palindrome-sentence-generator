"""Tests for entailment-linked endpoint confirmation generation."""

from experiments.entailment_linked_endpoint_confirmation_20260913 import (
    endpoint_automaton_inventory,
    endpoint_compatibility_automaton,
    independent_discourse_parse,
    independent_entailment_verify,
    productive_derivations,
    replay_character_ledger,
    run,
)


def test_semantic_witnesses_replay_independently_before_search():
    derivations = productive_derivations()
    result = run(state_limit=100_000)

    assert result["config"]["semantic_relation_selected_before_expansion"]
    assert result["config"]["machine_checkable_entailment_witness"]
    assert result["config"]["independent_entailment_replay"]
    assert result["derivation_count"] == len(derivations) == 20
    for derivation in derivations:
        parsed = independent_discourse_parse(derivation, derivation.rendered)
        parsed["evidence_rule"] = derivation.evidence_rule
        assert parsed["ok"]
        assert independent_entailment_verify(parsed)["ok"]


def test_endpoint_automaton_selects_only_cross_boundary_semantic_endpoints():
    inventory = endpoint_automaton_inventory()
    selected = endpoint_compatibility_automaton()
    result = run(state_limit=100_000)

    assert len(inventory) == 14
    assert len(selected) == 6
    assert {pair.operator for pair in selected} == {"can", "are", "do"}
    assert all(len(pair.shared_prefix) >= len(pair.operator) + 1 for pair in selected)
    assert all(not row["crosses_operator_subject_boundary"] for row in inventory if row["operator"] == "will")
    for left, right in zip(result["left_grammar_rows"], result["right_grammar_rows"]):
        assert right["reversed_suffix_tape"] == right["forward_suffix_tape"][::-1]
        assert left["evidence_rule"] == result["grammar_inventory"][[x["relation"] for x in result["grammar_inventory"]].index(left["relation"])] ["evidence_rule"]


def test_entailment_linked_search_persists_real_frontier_without_candidate():
    result = run(state_limit=100_000)
    failure = result["first_jointly_reachable_failure"]

    assert result["states_exhausted"] is True
    assert result["intersection"]["search_truncated"] is False
    assert result["intersection"]["states_examined"] == 12
    assert result["intersection"]["deepest_joint_state"]["prefix"] == "arem"
    assert failure == {
        "prefix": "arem",
        "left_next_chars": ["e", "u"],
        "right_next_chars": ["a"],
        "reason": "no_common_next_character",
    }
    assert result["records"] == []
    assert result["exact_candidates"] == []
    assert result["parsed_exact_candidates"] == []
    assert result["mechanically_admitted"] == []


def test_terminal_ledger_replay_and_truncation_are_fail_closed():
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
