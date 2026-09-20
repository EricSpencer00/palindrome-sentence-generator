from experiments.wh_question_chart_orbit_20260920 import audit,complete_derivations,consume,run
def test_wh_controls_have_typed_extraction():
    paths=complete_derivations();assert paths
    rendered=[" ".join(x.text for x in p) for p in paths]
    assert any("which letter" in x and "does" in x for x in rendered)
    assert any("what sign" in x for x in rendered)
def test_live_chart_and_audit():
    assert consume("abc","cba")==("","");assert consume("abc","ba")==("c","");assert consume("abc","xyz") is None
    result=run(state_limit=20_000);assert result["stats"]["grammar_paths"]>0
    for row in result["candidates"]:assert row["audit"]==audit(row["rendered"]);assert row["audit"]["exact"]
