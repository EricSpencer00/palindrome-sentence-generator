from experiments.shakespeare_scene_orbit_20260919 import audit, build_scene_lattice, run


def test_scene_lattice_is_authored_and_typed():
    lattice = build_scene_lattice()
    assert len(lattice) == 6
    assert all(frame.scene == "court" and frame.valency for slot in lattice for frame in slot)


def test_orbit_run_is_independently_audited():
    result = run(state_limit=30_000)
    assert result["provenance"]["human_authored_frames"]
    for candidate in result["candidates"]:
        assert candidate["audit"] == audit(candidate["rendered"])
        assert candidate["audit"]["exact"]
    assert result["provenance"]["finished_tape_reversal"] is False
    assert result["provenance"]["post_hoc_repair"] is False
