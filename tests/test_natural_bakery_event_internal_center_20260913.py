from experiments.natural_bakery_event_internal_center_20260913 import BASE_SOURCE_SHA256, run


def test_complete_bakery_event_frames_precede_center_selection():
    result = run(state_limit=5_000, closure_limit=10)
    frames = result["sentence_frame_inventory"]
    assert frames[0]["first_sentence"] == "The careful baker baked a fresh challah during the annual festival."
    assert frames[0]["second_sentence"] == "The careful baker recorded the family recipe after the local festival today."
    assert frames[0]["semantic_status"].startswith("complete ordinary")
    assert frames[1]["first_sentence"] == "The patient baker baked a fresh challah during the public festival."
    assert result["centre_inventory"][0]["word"] == "challah"
    assert result["centre_inventory"][0]["pivots"][3]["split"] == "chal|lah"
    assert result["centre_inventory"][0]["pivots"][3]["internal_prefix_matches"] == 3
    assert result["config"]["complete_event_pair_before_center_search"]
    assert result["config"]["pivot_derived_from_center_boundary"]
    assert result["config"]["prepared_multiword_center"] is False
    assert result["provenance"]["base_generator_sha256"] == BASE_SOURCE_SHA256


def test_bakery_search_replays_three_center_cancellations_before_rejection():
    result = run(state_limit=100_000, closure_limit=100)
    deepest = result["deepest_full_scheduler_replay"]
    assert deepest["independent_replay"]["events_replayed"] == len(deepest["ledger_before_rejection"])
    assert deepest["independent_replay"]["cancellations"] == 3
    assert deepest["next_literal_rejection"] == {
        "side": "left",
        "slot": 6,
        "word": "challah",
        "char": "c",
        "residual_before": "d",
        "action": "contradiction",
        "expected": "d",
    }
    assert deepest["independent_replay"]["ok"] is False
    assert all(row["independent_parse"] for row in result["complete_grammar_controls"])
    assert all(row["independent_exact_audit"]["letters"] > 100 for row in result["complete_grammar_controls"])
    assert result["exact_closures"] == []
    assert result["admitted_closures"] == []
