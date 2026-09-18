from experiments.sneak_attack_internal_center_variant_20260913 import run


def test_sneak_attack_is_the_only_attack_frame_and_is_independently_parsed():
    result = run(state_limit=5_000, closure_limit=10)
    attack = next(row for row in result["boundary_inventory"] if row["center"] == "attack")
    assert attack["adjacent_adjective"] == "sneak"
    assert attack["relation"] == "during"
    assert attack["trace"]["replay"]["cancellations"] == 2
    assert result["config"]["ordinary_sneak_attack_frame"]
    assert result["config"]["prior_career_artifact_untouched"]
    assert result["complete_grammar_controls"][0]["independent_parse"]


def test_sneak_attack_variant_persists_actual_deepest_scheduler_replay():
    result = run(state_limit=100_000, closure_limit=100)
    deepest = result["deepest_full_scheduler_replay"]
    assert deepest["ledger_before_rejection"]
    assert deepest["independent_replay"]["events_replayed"] == len(deepest["ledger_before_rejection"])
    assert deepest["next_literal_rejection"]["action"] == "contradiction"
    assert deepest["next_literal_rejection"]["word"] == "attack"
    assert deepest["next_literal_rejection"]["char"] == "c"
    assert deepest["next_literal_rejection"]["expected"] == "k"
    assert result["exact_closures"] == []
    assert result["admitted_closures"] == []
