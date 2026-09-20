from experiments.subordinate_conjunction_orbit_20260920 import audit, build_lattice, consume, run


def test_lattice_is_two_complete_clauses_with_bridge():
    lattice = build_lattice()
    assert [slot[0].role for slot in lattice] == ["subject", "event", "object", "subordinator", "object", "event", "subject"]
    assert lattice[3][0].clause == "bridge"
    assert lattice[4][0].clause == "subordinate"


def test_live_residual_and_independent_audit():
    assert consume("abc", "cba") == ("", "")
    assert consume("abc", "ba") == ("c", "")
    assert consume("abc", "xyz") is None
    result = run(state_limit=20_000)
    for row in result["candidates"]:
        assert row["audit"] == audit(row["rendered"])
        assert row["audit"]["exact"]
    assert result["provenance"]["agreement_state_before_emission"]
