import json,subprocess,sys
from pathlib import Path
R=Path(__file__).parents[1]
def test_contract():
 subprocess.run([sys.executable,str(R/'experiments/lexical_mirror_hypergraph_edge_repair_20260916.py')],check=True)
 d=json.loads((R/'runs/lexical-mirror-hypergraph-edge-repair-20260916.json').read_text());x=d['rows'][0];assert d['novelty_preflight']['passed'];assert x['repair_state']['single_state'];assert x['repair_state']['connected_role_graph'];assert x['audit']['sha256_equal'] is False;assert x['next_repair']
