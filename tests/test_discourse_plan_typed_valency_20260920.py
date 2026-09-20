from experiments.discourse_plan_typed_valency_20260920 import audit, consume, run

def test_typed_valencies_and_control_bound():
 d=run(1000); assert set(d["predicate_valencies"])=={"transitive","intransitive","ditransitive"}; assert d["control_count"]>=20

def test_live_equation():
 assert consume("abcd","abc")==("d","")
 assert consume("abcd","abx") is None

def test_rows_have_independent_audit():
 d=run(12000)
 for row in d["exact_candidates"]: assert row["audit"]==audit(row["rendered"])
