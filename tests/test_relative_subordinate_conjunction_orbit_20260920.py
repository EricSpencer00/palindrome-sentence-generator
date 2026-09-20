from experiments.relative_subordinate_conjunction_orbit_20260920 import audit, build_lattice, consume, run


def test_lattice_embeds_relative_clause_in_subordinate_subject():
    lattice = build_lattice()
    assert lattice[6][0].role == "relative_event"
    assert lattice[6][0].clause == "relative"
    assert lattice[3][0].role == "subordinator"
    assert lattice[4][0].clause == "subordinate"


def test_live_residual_and_independent_audit():
    assert consume("abc", "cba") == ("", "")
    assert consume("abc", "ba") == ("c", "")
    assert consume("abc", "xyz") is None
    result = run(state_limit=20_000)
    for row in result["candidates"]:
        assert row["audit"] == audit(row["rendered"])
        assert row["audit"]["exact"]
    assert result["provenance"]["embedded_relative_clause"]
