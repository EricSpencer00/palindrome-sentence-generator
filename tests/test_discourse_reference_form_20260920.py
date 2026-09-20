from experiments.discourse_reference_form_20260920 import audit, consume, run
def test_reference_controls_and_binding():
 d=run(1000); assert d["scenes"]==5 and d["control_count"]>=20
 assert any(c["reference_form"] in {"it","they"} for c in d["controls"])
def test_live_equation(): assert consume("abcd","abc")==("d","") and consume("abcd","abx") is None
def test_independent_audits():
 d=run(12000)
 for row in d["exact_candidates"]: assert row["audit"]==audit(row["rendered"])
