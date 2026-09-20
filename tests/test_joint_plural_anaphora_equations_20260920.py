from experiments.joint_plural_anaphora_equations_20260920 import audit,banks,consume,controls,run
def test_joint_plural_coreference_and_controls():
 assert banks("benefit")[0][0].number=="plural";assert banks("benefit")[6][0].anaphoric;assert banks("benefit")[10][0].anaphoric;assert len(controls())>=20;assert all(x["audit"]==audit(x["rendered"]) for x in controls())
def test_live_equation_audit():
 assert consume("abc","cba")==("","");assert consume("abc","ba")==("c","");assert consume("abc","xyz") is None
 result=run(state_limit=20000);assert result["stats"]["relations"]==2
 for row in result["candidates"]:assert row["audit"]==audit(row["rendered"]);assert row["audit"]["exact"]
