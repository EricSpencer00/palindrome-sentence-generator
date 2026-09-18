from experiments.natural_grammar_guidance_single_sentence_20260913 import run


def test_complete_guidance_frames_precede_internal_center_search():
    result = run(state_limit=5_000, closure_limit=10)
    frames = result["sentence_frame_inventory"]
    assert frames[0]["rendered"] == (
        "Clear grammar guides patient writers through a difficult exercise while "
        "careful editors revise complex technical prose today."
    )
    assert frames[0]["semantic_status"].startswith("complete ordinary")
    assert result["derived_center"] == {"word": "grammar", "pivot": 4, "split": "gram|mar", "initial_matches": 3}
    assert result["config"]["complete_event_frame_before_center_search"]
    assert result["config"]["post_center_predicate_semantically_selected"]
    assert result["config"]["no_repeated_content_topology"]
    assert result["config"]["prepared_multiword_center"] is False


def test_guidance_scheduler_replays_actual_predicate_frontier_and_controls_pass_hard_gates():
    result = run(state_limit=100_000, closure_limit=100)
    deepest = result["deepest_full_scheduler_replay"]
    assert deepest["independent_replay"]["events_replayed"] == len(deepest["ledger_before_rejection"])
    assert deepest["independent_replay"]["cancellations"] == 4
    assert deepest["next_literal_rejection"] == {
        "side": "right", "slot": 2, "word": "guides", "char": "u",
        "residual_before": "r", "action": "contradiction", "expected": "r",
    }
    assert deepest["independent_replay"]["ok"] is False
    assert all(row["independent_parse"] for row in result["complete_grammar_controls"])
    assert all(row["independent_exact_audit"]["letters"] > 100 for row in result["complete_grammar_controls"])
    assert all(row["central_admission"]["distinct_words"] for row in result["complete_grammar_controls"])
    assert all(row["central_admission"]["no_repeated_nontrivial_unit"] for row in result["complete_grammar_controls"])
    assert result["exact_closures"] == []
    assert result["admitted_closures"] == []
