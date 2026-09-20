from experiments.free_center_discourse_growth_20260920 import audit, boundary_compatible, controls, run, slots


def test_free_center_has_complete_controls_and_varied_pivots():
    assert len(slots()) == 7
    assert {slot[0].role for slot in slots()[1:-1]} >= {"verb", "object", "connector"}
    assert len(controls()) >= 20
    assert all(row["audit"] == audit(row["rendered"]) for row in controls())


def test_center_boundary_and_independent_audit():
    assert boundary_compatible("abc", "cba")
    assert not boundary_compatible("abc", "xyz")
    result = run(state_limit=20_000)
    assert result["stats"]["pivot_positions"] == 5
    for row in result["candidates"]:
        assert row["audit"] == audit(row["rendered"])
        assert row["audit"]["exact"]
