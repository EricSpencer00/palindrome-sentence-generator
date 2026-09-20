from experiments.typed_recipient_dative_equations_20260920 import audit, banks, consume, controls, run


def test_recipient_theme_roles_and_controls():
    lattice = banks()
    assert len(lattice) == 9
    assert lattice[2][0].role == "recipient"
    assert lattice[3][0].role == "theme"
    assert all(len(item.text.split()) == 2 for item in lattice[2] + lattice[3] + lattice[7] + lattice[8])
    assert len(controls()) >= 20
    assert all(row["audit"] == audit(row["rendered"]) for row in controls())


def test_live_equation_and_independent_audit():
    assert consume("abc", "cba") == ("", "")
    assert consume("abc", "ba") == ("c", "")
    assert consume("abc", "xyz") is None
    result = run(state_limit=20_000)
    assert result["stats"]["states"] > 0
    for row in result["candidates"]:
        assert row["audit"] == audit(row["rendered"])
        assert row["audit"]["exact"]
