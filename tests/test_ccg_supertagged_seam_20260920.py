from experiments.ccg_supertagged_seam_20260920 import audit, combine, consume, run

def test_ccg_application_and_composition_exist():
 from experiments.ccg_supertagged_seam_20260920 import S,NP,TV
 assert combine(TV,NP)[1]=="forward_application"

def test_live_equation():
 assert consume("abcd","abc")==("d","")
 assert consume("abcd","abx") is None

def test_rows_are_independently_audited():
 d=run(12000)
 for row in d["exact_candidates"]: assert row["audit"]==audit(row["rendered"])
