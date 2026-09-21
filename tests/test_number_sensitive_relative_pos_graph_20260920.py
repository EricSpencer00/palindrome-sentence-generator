import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/"experiments/number_sensitive_relative_pos_graph_20260920.py"
s=importlib.util.spec_from_file_location("relative_graph",P); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
def test_relative_graph():
 r=m.run()
 assert r["stats"]["controls"]==8
 assert r["stats"]["exact_gt38"]==0
 assert all(x["audit"]["two_pointer_checked"] and x["anti_shortcut"]["word_order_symmetry"] is False for x in r["diagnostic_controls"])
