from experiments.cross_paired_clause_orbit_20260920 import audit, build_lattice, consume, run


def test_cross_pair_has_complete_permuted_clauses_and_heldout_frames():
    lattice = build_lattice()
    assert len(lattice) == 8
    assert [slot[0].role for slot in lattice] == ["subject", "event", "object", "setting", "setting", "object", "event", "subject"]
    assert all(slot[0].split == "B" for slot in lattice[4:])


def test_residual_consumption_and_independent_audit():
    assert consume("abc", "cba") == ("", "")
    assert consume("abc", "ba") == ("c", "")
    assert consume("a", "cba") == ("", "cb")
    assert consume("abc", "xyz") is None
    result = run(state_limit=20_000)
    for row in result["candidates"]:
        assert row["audit"] == audit(row["rendered"])
        assert row["audit"]["exact"]
    assert result["provenance"]["finished_tape_reversal"] is False
