from experiments.bridge_complement_relative_orbit_20260920 import audit, build_lattice, consume, run


def test_bridge_selects_finite_complement_and_relative_is_typed():
    lattice = build_lattice()
    assert lattice[3][0].role == "bridge"
    assert lattice[3][0].complement == "finite"
    assert lattice[6][0].role == "relative"
    assert lattice[6][0].valency == "subject-modifier"


def test_live_residual_and_independent_audit():
    assert consume("abc", "cba") == ("", "")
    assert consume("abc", "ba") == ("c", "")
    assert consume("abc", "xyz") is None
    result = run(state_limit=20_000)
    for row in result["candidates"]:
        assert row["audit"] == audit(row["rendered"])
        assert row["audit"]["exact"]
    assert result["provenance"]["bridge_complement_selected_before_expansion"]
