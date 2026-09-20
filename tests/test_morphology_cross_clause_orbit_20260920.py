from experiments.morphology_cross_clause_orbit_20260920 import audit, build_lattice, consume, run


def test_lattice_has_cross_roles_and_morphological_states():
    lattice = build_lattice()
    assert [slot[0].role for slot in lattice] == ["subject", "event", "object", "setting", "setting", "object", "event", "subject"]
    assert lattice[0][0].number == "singular"
    assert lattice[7][0].number == "plural"
    assert any(frame.agreement == "plural" for frame in lattice[6])


def test_consumption_and_audit_are_independent():
    assert consume("abc", "cba") == ("", "")
    assert consume("abc", "ba") == ("c", "")
    assert consume("abc", "xyz") is None
    result = run(state_limit=20_000)
    for row in result["candidates"]:
        assert row["audit"] == audit(row["rendered"])
        assert row["audit"]["exact"]
    assert result["provenance"]["agreement_state_carried"]
