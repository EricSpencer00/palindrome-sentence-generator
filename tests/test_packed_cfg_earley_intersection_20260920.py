from experiments.packed_cfg_earley_intersection_20260920 import audit, consume, packed, paths, run

def test_cfg_has_recursive_complete_frames():
 assert packed("S",3)
 assert any("CONJ" in p or "REL" in p for p in paths(3))

def test_residual_pointer():
 assert consume("abcd","abc")==("d","")
 assert consume("abcd","abx") is None

def test_controls_and_independent_audits():
 d=run(12000,3); assert d["compatible_seeds"]>0 and d["controls"]
 for x in d["exact_candidates"]: assert x["audit"]==audit(x["rendered"])
