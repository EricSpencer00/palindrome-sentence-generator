from experiments.natural_tinnitus_clinical_single_sentence_20260913 import run


def test_complete_clinical_frame_precedes_internal_center_search():
    result = run(state_limit=5_000, closure_limit=10)
    assert result["sentence_frame_inventory"][0]["rendered"] == (
        "Chronic tinnitus causes persistent sleep loss while careful clinicians "
        "investigate underlying hearing damage in a quiet clinic today."
    )
    assert result["sentence_frame_inventory"][0]["semantic_status"].startswith("complete ordinary")
    assert result["derived_center"] == {"word": "tinnitus", "pivot": 3, "split": "tin|nitus", "initial_matches": 3}
    assert result["config"]["complete_event_frame_before_center_search"]
    assert result["config"]["no_repeated_content_topology"]
    assert result["config"]["post_center_predicate_semantically_selected"]
    assert result["config"]["prepared_multiword_center"] is False


def test_clinical_scheduler_replays_actual_frontier_and_controls_pass_hard_gates():
    result = run(state_limit=100_000, closure_limit=100)
    deepest = result["deepest_full_scheduler_replay"]
    assert deepest["independent_replay"]["events_replayed"] == len(deepest["ledger_before_rejection"])
    assert deepest["independent_replay"]["cancellations"] == 3
    assert deepest["next_literal_rejection"] == {
        "side": "left", "slot": 0, "word": "chronic", "char": "c",
        "residual_before": "u", "action": "contradiction", "expected": "u",
    }
    assert deepest["independent_replay"]["ok"] is False
    assert all(row["independent_parse"] for row in result["complete_grammar_controls"])
    assert all(row["independent_exact_audit"]["letters"] > 100 for row in result["complete_grammar_controls"])
    assert all(row["central_admission"]["distinct_words"] for row in result["complete_grammar_controls"])
    assert all(row["central_admission"]["no_repeated_nontrivial_unit"] for row in result["complete_grammar_controls"])
    assert result["exact_closures"] == []
    assert result["admitted_closures"] == []
