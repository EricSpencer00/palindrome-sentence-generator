from experiments.complement_coordination_hypergraph_20260920 import audit, consume, run

def test_hypergraph_topologies_and_controls():
 d=run(1000); assert d["hypergraphs"]==4 and d["topology_pairs"]==16 and d["compatible_seeds"]>0

def test_live_character_equation():
 assert consume("abcd","abc")==("d","")
 assert consume("abcd","abx") is None

def test_exact_rows_are_independently_audited():
 d=run(12000)
 for row in d["exact_candidates"]: assert row["audit"]==audit(row["rendered"])
