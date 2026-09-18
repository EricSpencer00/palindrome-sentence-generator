from experiments.centerout_event_chart_20260913 import EVENT, run


def test_event_is_fixed_before_search_and_seed_is_intact_prose():
    result = run(state_limit=20_000)
    assert result["config"]["event_fixed_before_search"]
    assert result["config"]["two_boundary_lexical_chart"]
    assert result["config"]["sentence_order_deferred"]
    assert result["seed_control"]["rendered"] == "A calm baker serves warm bread to kind neighbors."
    assert result["seed_control"]["independent_parse"]
    assert result["seed_control"]["independent_exact_audit"]["letters"] >= 30
    assert result["reader_status"].startswith("unreviewed")


def test_chart_records_replayable_failure_and_does_not_promote_controls():
    result = run(state_limit=100_000)
    frontier = result["deepest_live_frontier"]
    assert frontier["independent_replay"]["events_replayed"] == len(frontier["ledger"])
    assert frontier["rejection"] is not None
    assert result["exact_closures"] == []
    assert result["admitted_closures"] == []
    assert result["seed_control"]["mechanically_admitted"] is False


def test_event_roles_remain_explicit():
    assert EVENT["subject"] == ("a", "calm", "baker")
    assert EVENT["predicate"] == ("serves",)
    assert EVENT["object"] == ("warm", "bread")
    assert EVENT["recipient"] == ("to", "kind", "neighbors")
