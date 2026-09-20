from experiments.discourse_plan_typed_valency_plain_20260920 import audit, consume, run
def test_plain_controls_and_valencies():
 d=run(1000); assert d["control_count"]>=20 and set(d["predicate_valencies"])=={"transitive","intransitive","ditransitive"}
 assert all(d["max_letters"]>=c["audit"]["letters"] for c in d["controls"])
def test_live_equation():
 assert consume("abcd","abc")==("d","") and consume("abcd","abx") is None
def test_audits():
 d=run(12000)
 for row in d["exact_candidates"]: assert row["audit"]==audit(row["rendered"])
