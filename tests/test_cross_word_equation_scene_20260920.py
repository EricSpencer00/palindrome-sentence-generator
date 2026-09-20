from experiments.cross_word_equation_scene_20260920 import audit,banks,consume,controls,run
def test_equation_banks_and_controls():
    assert len(banks())==7;assert len(controls())>=20;assert all(x["audit"]==audit(x["rendered"]) for x in controls())
def test_live_equations_and_audit():
    assert consume("abc","cba")==("","");assert consume("abc","ba")==("c","");assert consume("abc","xyz") is None
    result=run(state_limit=20_000);assert result["stats"]["states"]>0
    for row in result["candidates"]:assert row["audit"]==audit(row["rendered"]);assert row["audit"]["exact"]
