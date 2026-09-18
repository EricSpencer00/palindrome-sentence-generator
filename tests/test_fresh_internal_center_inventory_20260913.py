from experiments.fresh_internal_center_inventory_20260913 import run


def test_inventory_has_real_ordinary_internal_pivots():
    result = run(state_limit=5_000, closure_limit=10)
    inventory = {row["word"]: row for row in result["centre_inventory"]}
    assert "career" not in inventory
    assert inventory["attack"]["pivots"][1]["split"] == "at|tack"
    assert inventory["attack"]["pivots"][1]["internal_prefix_matches"] == 2
    assert inventory["effect"]["pivots"][1]["split"] == "ef|fect"
    assert result["config"]["prepared_multiword_center"] is False


def test_joint_boundary_inventory_selects_a_semantic_single_word_frame():
    result = run(state_limit=5_000, closure_limit=10)
    best = result["best_boundary_trace"]
    assert (best["center"], best["pivot"], best["adjacent_adjective"], best["relation"]) == ("attack", "at|tack", "public", "during")
    assert best["trace"]["replay"]["cancellations"] == 3
    assert best["trace"]["first_incompatible"] == {"side": "right", "word": "attack", "char": "k", "expected": "i", "residual": "i"}
    assert result["stats"]["semantic_boundary_triples_enumerated"] == 3


def test_fresh_inventory_runs_the_complete_tree_scheduler_and_parses_controls():
    result = run(state_limit=100_000, closure_limit=100)
    assert result["stats"]["character_emissions"] == result["stats"]["state_count"]
    assert result["stats"]["lexical_assignments_considered"] > 1
    assert result["stats"]["residual_cancellations"] > 0
    assert result["exact_closures"] == []
    assert result["admitted_closures"] == []
    for control in result["complete_grammar_controls"]:
        assert control["independent_parse"]
        assert control["independent_exact_audit"]["letters"] > 100
        assert control["central_admission"]["no_self_palindromic_proper_multiword_span"]


def test_deepest_ledger_is_from_scheduler_and_replays_before_rejection():
    result = run(state_limit=100_000, closure_limit=100)
    deepest = result["deepest_full_scheduler_replay"]
    assert deepest["ledger_before_rejection"]
    assert deepest["next_literal_rejection"]["action"] == "contradiction"
    assert deepest["independent_replay"]["events_replayed"] == len(deepest["ledger_before_rejection"])
    assert deepest["next_literal_rejection"]["expected"] != deepest["next_literal_rejection"]["char"]
    assert result["admitted_closures"] == []
