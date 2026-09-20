from experiments.contrastive_relation_equations_20260920 import audit,banks,consume,controls,run
def test_contrastive_state_selects_connective():
 assert banks("benefit")[5][0].text=="although";assert banks("transfer")[5][0].text=="while";assert banks("benefit")[3][0].text=="for";assert banks("transfer")[3][0].text=="to";assert len(controls())>=20;assert all(x["audit"]==audit(x["rendered"]) for x in controls())
def test_live_equation_audit():
 assert consume("abc","cba")==("","");assert consume("abc","ba")==("c","");assert consume("abc","xyz") is None
 result=run(state_limit=20000);assert result["stats"]["relations"]==2
 for row in result["candidates"]:assert row["audit"]==audit(row["rendered"]);assert row["audit"]["exact"]
