from experiments.recursive_series_typed_relative_20260920 import audit, decorate, paths, run

def test_relative_paths_are_complete():
 assert any("REL" in p for p in paths())
 assert all(p[-1]=="VP" for p in paths())

def test_punctuation_is_editorial():
 x=decorate("the baker who greets the child","the baker")
 assert "who" in x and audit(x)["letters"]==audit("the baker who greets the child; the baker")["letters"]

def test_run_audit():
 d=run(12000); assert d["compatible_seeds"]>0 and d["controls"]
 for row in d["exact_candidates"]: assert row["audit"]==audit(row["rendered"])
