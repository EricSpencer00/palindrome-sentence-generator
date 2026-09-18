from experiments.internal_pivot_adjacent_event_repair_20260913 import (
    Counter,
    Grammar,
    Symbol,
    diagnostic_trace,
    run,
    slots,
)


def test_center_is_a_grammar_leaf_and_neighbors_are_real_event_slots():
    leaf_slots = slots(Grammar().expand(Symbol("S")))
    center = next(slot.index for slot in leaf_slots if slot.symbol.role == "center_word")
    assert center == 6
    assert leaf_slots[center].symbol.kind == "noun_artifact"
    assert leaf_slots[center - 1].symbol.role == "s1_verb"
    assert leaf_slots[center + 1].symbol.role == "during"


def test_internal_trace_reaches_the_repaired_deeper_frontier():
    stats = Counter()
    found = diagnostic_trace(stats)
    trace = found["trace"]
    assert [event["char"] for event in trace["ledger"]] == list("eetthhsd")
    assert trace["replay"]["cancellations"] == 3
    assert trace["first_incompatible"] == {
        "side": "right", "word": "during", "char": "d", "expected": "s", "residual": "s"
    }
    repaired = found["temporal_repair"]["trace"]
    assert found["temporal_repair"]["words"] == ("polish", "teeth", "since")
    assert [event["char"] for event in repaired["ledger"]] == list("eetthhssiiln")
    assert repaired["replay"]["cancellations"] == 5
    assert repaired["first_incompatible"]["word"] == "since"
    assert repaired["first_incompatible"]["char"] == "n"
    assert repaired["first_incompatible"]["expected"] == "l"
    assert len(found["boundary_triple_enumeration"]) == 9


def test_typed_search_emits_real_tree_characters_and_rejects_no_shortcut_candidate():
    result = run(state_limit=100_000, closure_limit=100)
    stats = result["stats"]
    assert result["config"]["one_connected_tree"]
    assert result["config"]["grammar_owns_every_leaf"]
    assert result["config"]["one_character_emission_states"]
    assert result["config"]["reject_every_self_palindromic_contiguous_multiword_span"]
    assert stats["lexical_assignments_considered"] > 1
    assert stats["character_emissions"] == stats["state_count"]
    assert stats["residual_cancellations"] > 0
    assert stats["temporal_repair_trace_cancellations"] == 5
    assert stats["typed_boundary_triples_enumerated"] == 9
    assert result["exact_closures"] == []
    assert result["admitted_closures"] == []
    for control in result["complete_grammar_controls"]:
        assert control["independent_parse"]
        assert control["independent_exact_audit"]["letters"] > 100
        assert control["central_admission"]["no_self_palindromic_proper_multiword_span"]
