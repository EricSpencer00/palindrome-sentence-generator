from experiments.discourse_plan_tense_aspect_20260920 import audit, consume, run, surface
def test_morphology_and_controls():
 d=run(1000); assert d["control_count"]>=20 and surface("notice","third","present","simple")=="notices" and "noticing" in surface("notice","base","present","progressive")
def test_live_equation(): assert consume("abcd","abc")==("d","") and consume("abcd","abx") is None
def test_audits():
 d=run(12000)
 for row in d["exact_candidates"]: assert row["audit"]==audit(row["rendered"])
