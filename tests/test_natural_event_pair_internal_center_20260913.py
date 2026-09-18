from experiments.natural_event_pair_internal_center_20260913 import BASE_SOURCE_SHA256, run


def test_complete_natural_event_pairs_precede_center_inventory():
    result = run(state_limit=5_000, closure_limit=10)
    frames = result["sentence_frame_inventory"]
    assert frames[0]["first_sentence"] == "The careful researcher planned a brief meeting during the annual study."
    assert frames[0]["second_sentence"] == "The careful researcher documented the annual report after the public meeting today."
    assert frames[0]["semantic_status"].startswith("complete ordinary")
    assert frames[1]["first_sentence"] == "The patient analyst planned a brief meeting during the public review."
    assert result["centre_inventory"][0]["word"] == "meeting"
    assert result["centre_inventory"][0]["pivots"][1]["split"] == "me|eting"
    assert result["config"]["complete_event_pair_before_center_search"]
    assert result["config"]["frame_first_inventory"]
    assert result["config"]["prepared_multiword_center"] is False
    assert result["provenance"]["base_generator_sha256"] == BASE_SOURCE_SHA256


def test_natural_pair_search_keeps_exact_admission_exact_only_and_replays_frontier():
    result = run(state_limit=100_000, closure_limit=100)
    deepest = result["deepest_full_scheduler_replay"]
    assert deepest["independent_replay"]["events_replayed"] == len(deepest["ledger_before_rejection"])
    assert deepest["next_literal_rejection"] == {
        "side": "left",
        "slot": 6,
        "word": "meeting",
        "char": "m",
        "residual_before": "t",
        "action": "contradiction",
        "expected": "t",
    }
    assert deepest["independent_replay"]["ok"] is False
    assert all(row["independent_parse"] for row in result["complete_grammar_controls"])
    assert all(row["independent_exact_audit"]["letters"] > 100 for row in result["complete_grammar_controls"])
    assert result["exact_closures"] == []
    assert result["admitted_closures"] == []
