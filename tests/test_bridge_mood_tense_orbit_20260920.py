from experiments.bridge_mood_tense_orbit_20260920 import audit, build_lattice, consume, run


def test_bridge_carries_mood_and_tense_state():
    lattice = build_lattice()
    assert lattice[3][0].role == "bridge"
    assert lattice[3][0].mood == "indicative"
    assert lattice[3][0].tense == "present"
    assert lattice[5][0].role == "event"


def test_live_residual_and_independent_audit():
    assert consume("abc", "cba") == ("", "")
    assert consume("abc", "ba") == ("c", "")
    assert consume("abc", "xyz") is None
    result = run(state_limit=20_000)
    for row in result["candidates"]:
        assert row["audit"] == audit(row["rendered"])
        assert row["audit"]["exact"]
    assert result["provenance"]["bridge_state_before_expansion"]
