from experiments.typed_object_chunk_equations_20260920 import audit,banks,consume,controls,run
def test_object_chunks_carry_features_and_controls_are_intact():
 assert len(banks())==7;assert all(len(x.text.split())==2 for x in banks()[2]+banks()[6]);assert len(controls())>=20;assert all(x["audit"]==audit(x["rendered"]) for x in controls())
def test_live_equation_audit():
 assert consume("abc","cba")==("","");assert consume("abc","ba")==("c","");assert consume("abc","xyz") is None
 result=run(state_limit=20000);assert result["stats"]["states"]>0
 for row in result["candidates"]:assert row["audit"]==audit(row["rendered"]);assert row["audit"]["exact"]
