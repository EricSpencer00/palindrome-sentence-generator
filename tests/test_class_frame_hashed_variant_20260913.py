from experiments.class_frame_hashed_variant_20260913 import BASE_SOURCE_SHA256, run


def test_frame_first_inventory_owns_an_ordinary_class_center():
    result = run(state_limit=5_000, closure_limit=10)
    assert result["sentence_frame_inventory"][0]["surface"] == "held a small class along the western route"
    assert result["centre_inventory"][0]["word"] == "class"
    assert result["centre_inventory"][0]["pivots"][3]["split"] == "clas|s"
    assert result["config"]["frame_first_inventory"]
    assert result["config"]["prepared_multiword_center"] is False
    assert result["provenance"]["base_generator_sha256"] == BASE_SOURCE_SHA256


def test_class_frame_uses_actual_scheduler_replay_and_independent_controls():
    result = run(state_limit=100_000, closure_limit=100)
    deepest = result["deepest_full_scheduler_replay"]
    assert deepest["independent_replay"]["events_replayed"] == len(deepest["ledger_before_rejection"])
    assert deepest["next_literal_rejection"]["action"] == "contradiction"
    assert deepest["next_literal_rejection"]["word"] == "class"
    assert deepest["next_literal_rejection"]["char"] == "c"
    assert deepest["next_literal_rejection"]["expected"] == "o"
    assert all(row["independent_parse"] for row in result["complete_grammar_controls"])
    assert all(row["independent_exact_audit"]["letters"] > 100 for row in result["complete_grammar_controls"])
    assert result["exact_closures"] == []
    assert result["admitted_closures"] == []
