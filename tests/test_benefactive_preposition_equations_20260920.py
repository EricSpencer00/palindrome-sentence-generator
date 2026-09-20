from experiments.benefactive_preposition_equations_20260920 import audit,banks,consume,controls,run
def test_benefactive_frame_and_controls():
 lattice=banks();assert len(lattice)==11;assert lattice[3][0].role=="preposition";assert lattice[4][0].role=="recipient";assert len(controls())>=20;assert all(x["audit"]==audit(x["rendered"]) for x in controls())
def test_live_equation_audit():
 assert consume("abc","cba")==("","");assert consume("abc","ba")==("c","");assert consume("abc","xyz") is None
 result=run(state_limit=20000);assert result["stats"]["states"]>0
 for row in result["candidates"]:assert row["audit"]==audit(row["rendered"]);assert row["audit"]["exact"]
