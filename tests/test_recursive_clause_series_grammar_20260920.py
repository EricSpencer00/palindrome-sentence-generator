from experiments.recursive_clause_series_grammar_20260920 import audit, consume, recursive_paths, run

def test_recursive_paths_are_complete_and_variable():
 p=recursive_paths(3)
 assert [len(x) for x in p]==[2,5,8]
 assert all(x[-1]=="VP" for x in p)

def test_residual_consumption():
 assert consume("abcd","abc")==("d","")
 assert consume("abcd","abx") is None

def test_controls_and_audits():
 d=run(12000,3); assert d["compatible_seeds"]>0 and d["controls"]
 for row in d["exact_candidates"]: assert row["audit"]==audit(row["rendered"])
