import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/"experiments/agreement_adjunct_pos_graph_20260920.py"
s=importlib.util.spec_from_file_location("agreement_graph",P)
m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
def test_agreement_adjunct_graph():
 r=m.run()
 assert r["stats"]["controls"]==8
 assert r["stats"]["exact_gt38"]==0
 assert all(x["agreement"] in ("sg","pl") and x["audit"]["two_pointer_checked"] for x in r["diagnostic_controls"])
