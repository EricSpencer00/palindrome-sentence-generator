import json,subprocess,sys
from pathlib import Path
R=Path(__file__).parents[1]
def test_contract():
 subprocess.run([sys.executable,str(R/'experiments/scalable_outsidein_single_edge_repair3_20260916.py')],check=True)
 d=json.loads((R/'runs/scalable-outsidein-single-edge-repair3-20260916.json').read_text());x=d['rows'][0];assert d['summary']['max_length']>38;assert x['repair_state']['child_count']==1;assert x['audit']['sha256_equal'] is False;assert x['next_repair']
