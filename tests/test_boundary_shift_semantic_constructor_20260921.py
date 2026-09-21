import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/"experiments/boundary_shift_semantic_constructor_20260921.py"
s=importlib.util.spec_from_file_location("m",P); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
def test_audit():
 a=m.audit("An aide rips nine memos; some men inspire Diana"); assert a["pointer_exact"] and a["sha256_forward"]==a["sha256_reverse"]
def test_edges():
 idx=m.edge_index(); assert idx and all("word" in x for xs in idx.values() for x in xs)
def test_records_boundaries():
 f,t,c=m.candidates(2000); assert t==2001 and c and not f and all("boundary_lengths" in x for x in c)
