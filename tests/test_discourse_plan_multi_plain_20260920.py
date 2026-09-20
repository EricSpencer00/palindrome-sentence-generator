from experiments.discourse_plan_multi_plain_20260920 import audit, consume, run, surface
def test_eight_plans_and_controls():
 d=run(1000); assert d["plans"]==8 and d["control_count"]>=20 and max(c["audit"]["letters"] for c in d["controls"])<=80
def test_morphology():
 assert surface("send","third","past","simple")=="sent" and surface("notice","third","past","simple")=="noticed" and surface("notice","first","present","progressive")=="am noticing" and surface("notice","first","past","progressive")=="was noticing" and surface("notice","plural","past","progressive")=="were noticing" and surface("send","first","present","progressive")=="am sending"
def test_equations_and_audits():
 assert consume("abcd","abc")==("d","") and consume("abcd","abx") is None
 d=run(12000)
 for row in d["exact_candidates"]: assert row["audit"]==audit(row["rendered"])
