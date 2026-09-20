from experiments.blank_verse_parallel_scene_20260920 import audit,build_lattice,consume,run
def test_parallel_scene_roles_are_independent():
    lattice=build_lattice();assert [x[0].role for x in lattice]==["vocative","verb","adjective","noun","noun","adjective","verb","subject"]
    assert any(x.utterance=="A" for slot in lattice for x in slot);assert any(x.utterance=="B" for slot in lattice for x in slot)
def test_live_residual_and_audit():
    assert consume("abc","cba")==("","");assert consume("abc","ba")==("c","");assert consume("abc","xyz") is None
    result=run(state_limit=20_000);assert result["stats"]["states"]>0
    for row in result["candidates"]:assert row["audit"]==audit(row["rendered"]);assert row["audit"]["exact"]
