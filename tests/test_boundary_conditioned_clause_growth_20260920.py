from experiments.boundary_conditioned_clause_growth_20260920 import audit, consume, frames, run

def test_outer_boundary_seed_and_live_debt():
 assert consume("a","anaid")==("","naid")
 assert consume("a","river") is None

def test_frames_are_complete_and_variable():
 fs=frames(); assert len({len(x) for x in fs})>1
 assert all(x[-1] in {"NAME","OBJ"} for x in fs)

def test_controls_and_exact_rows_are_independently_audited():
 d=run(12000); assert d["boundary_seeds"]>0 and d["controls"]
 for x in d["exact_candidates"]: assert x["audit"]==audit(x["rendered"])
