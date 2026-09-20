from experiments.dual_relative_valency_orbit_20260920 import audit, build_lattice, consume, run


def test_two_relative_sites_have_distinct_valencies():
    lattice = build_lattice()
    assert lattice[1][0].role == "relative_subject"
    assert lattice[1][0].valency == "subject-modifier"
    assert lattice[5][0].role == "relative_object"
    assert lattice[5][0].valency == "object-modifier"


def test_live_residual_and_independent_audit():
    assert consume("abc", "cba") == ("", "")
    assert consume("abc", "ba") == ("c", "")
    assert consume("abc", "xyz") is None
    result = run(state_limit=20_000)
    for row in result["candidates"]:
        assert row["audit"] == audit(row["rendered"])
        assert row["audit"]["exact"]
    assert result["provenance"]["two_relative_attachment_sites"]
