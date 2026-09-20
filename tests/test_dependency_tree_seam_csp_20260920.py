from experiments.dependency_tree_seam_csp_20260920 import Tree, audit, consume, run

def test_tree_edges_are_typed():
 d=run(1000); assert d["trees"]==3 and d["compatible_seeds"]>0

def test_character_equation_consumption():
 assert consume("abcd","abc")==("d","")
 assert consume("abcd","abx") is None

def test_exact_rows_have_independent_audit():
 d=run(12000)
 for row in d["exact_candidates"]: assert row["audit"]==audit(row["rendered"])
