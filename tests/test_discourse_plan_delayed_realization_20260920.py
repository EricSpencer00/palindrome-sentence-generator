from experiments.discourse_plan_delayed_realization_20260920 import audit, consume, run

def test_plan_count_bounds_and_controls():
 d=run(1000); assert 8<=d["plans"]<=12 and d["control_count"]>=20

def test_live_character_equation():
 assert consume("abcd","abc")==("d","")
 assert consume("abcd","abx") is None

def test_exact_rows_have_independent_audit():
 d=run(12000)
 for row in d["exact_candidates"]: assert row["audit"]==audit(row["rendered"])
