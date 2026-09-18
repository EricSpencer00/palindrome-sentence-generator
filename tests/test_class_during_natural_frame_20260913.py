from experiments.class_during_natural_frame_20260913 import BASE_SOURCE_SHA256, run


def test_semantic_first_during_frame_is_fixed_before_search():
    result = run(state_limit=5_000, closure_limit=10)
    frame = result["sentence_frame_inventory"][0]
    assert frame["surface"] == "held a small class during the annual study"
    assert frame["relation"] == "during"
    assert frame["semantic_status"].startswith("ordinary")
    assert result["centre_inventory"][0]["word"] == "class"
    assert result["centre_inventory"][0]["pivots"][3]["split"] == "clas|s"
    assert result["config"]["semantic_frame_before_character_search"]
    assert result["config"]["frame_first_inventory"]
    assert result["config"]["prepared_multiword_center"] is False
    assert result["provenance"]["base_generator_sha256"] == BASE_SOURCE_SHA256


def test_during_frame_records_actual_replay_and_exact_only_admission():
    result = run(state_limit=100_000, closure_limit=100)
    deepest = result["deepest_full_scheduler_replay"]
    assert deepest["independent_replay"]["events_replayed"] == len(deepest["ledger_before_rejection"])
    assert deepest["next_literal_rejection"] == {
        "side": "left",
        "slot": 6,
        "word": "class",
        "char": "a",
        "residual_before": "d",
        "action": "contradiction",
        "expected": "d",
    }
    assert deepest["independent_replay"]["ok"] is False
    assert all(row["independent_parse"] for row in result["complete_grammar_controls"])
    assert all(row["independent_exact_audit"]["letters"] > 100 for row in result["complete_grammar_controls"])
    assert result["exact_closures"] == []
    assert result["admitted_closures"] == []
