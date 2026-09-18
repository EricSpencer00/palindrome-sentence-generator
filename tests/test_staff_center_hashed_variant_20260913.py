from experiments.staff_center_hashed_variant_20260913 import BASE_SOURCE_SHA256, run


def test_staff_is_a_fresh_single_word_center_with_hashed_provenance():
    result = run(state_limit=5_000, closure_limit=10)
    assert [(row["word"], row["pivots"][3]["split"]) for row in result["centre_inventory"]] == [("staff", "staf|f")]
    assert result["config"]["prepared_multiword_center"] is False
    assert result["config"]["prior_sources_hash_enforced"] is True
    assert result["provenance"]["base_generator_sha256"] == BASE_SOURCE_SHA256


def test_staff_frame_is_independently_reparsed_and_has_actual_scheduler_rejection():
    result = run(state_limit=100_000, closure_limit=100)
    control = result["complete_grammar_controls"][0]
    assert control["independent_parse"]
    assert control["independent_exact_audit"]["letters"] > 100
    deepest = result["deepest_full_scheduler_replay"]
    assert deepest["independent_replay"]["events_replayed"] == len(deepest["ledger_before_rejection"])
    assert deepest["next_literal_rejection"]["action"] == "contradiction"
    assert deepest["next_literal_rejection"]["word"] == "staff"
    assert deepest["next_literal_rejection"]["expected"] == "a"
    assert deepest["next_literal_rejection"]["char"] == "s"
    assert result["exact_closures"] == []
    assert result["admitted_closures"] == []
