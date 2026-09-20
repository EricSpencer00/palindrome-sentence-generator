from experiments.role_permuted_scene_graph_20260920 import audit, consume, run

def test_scene_graph_lane_has_independent_orders():
 d=run(1000); assert d["scene_graphs"]==3 and d["role_orders"]>3 and d["compatible_seeds"]>0

def test_live_equation():
 assert consume("abcd","abc")==("d","")
 assert consume("abcd","abx") is None

def test_rows_have_independent_audit():
 d=run(12000)
 for row in d["exact_candidates"]: assert row["audit"]==audit(row["rendered"])
