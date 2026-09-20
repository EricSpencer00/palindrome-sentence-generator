from experiments.agreement_subject_chunk_equations_20260920 import audit,banks,consume,controls,run
def test_subject_chunks_and_controls_are_grammatical():
    assert len(banks())==7;assert all(len(x.text.split())==2 for x in banks()[0]+banks()[4]);assert len(controls())>=20;assert all(x["audit"]==audit(x["rendered"]) for x in controls())
def test_live_equation_audit():
    assert consume("abc","cba")==("","");assert consume("abc","ba")==("c","");assert consume("abc","xyz") is None
    result=run(state_limit=20_000);assert result["stats"]["states"]>0
    for row in result["candidates"]:assert row["audit"]==audit(row["rendered"]);assert row["audit"]["exact"]
