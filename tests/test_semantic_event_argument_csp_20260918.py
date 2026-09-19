from experiments.semantic_event_argument_csp_20260918 import audit, build_events, run


def test_event_bank_carries_typed_roles_and_varied_settings():
    events = build_events()
    assert len(events) > 1000
    assert {event.place for event in events} >= {"", "at dawn", "in town", "near home", "by the sea", "in an arena"}
    assert all(event.actor and event.verb and event.obj for event in events)


def test_bounded_run_stops_at_probe_budget_and_audits_exact_rows():
    result = run(max_probes=120)
    assert result["stats"]["pair_worlds_checked"] == 120
    assert result["stats"]["stored_probes"] == 120
    assert result["stats"]["bounded"] is True
    assert result["stats"]["new_mechanically_admitted"] == 0
    for row in result["exact_candidates"]:
        assert row["audit"]["two_pointer_exact"] is True
        assert row["audit"]["sha_equal_under_reversal"] is True


def test_independent_audit_rejects_a_single_mismatch():
    result = audit("an aide reads a note")
    assert result["two_pointer_exact"] is False
    assert result["mismatch_count"] > 0
