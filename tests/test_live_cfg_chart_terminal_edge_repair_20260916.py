import json,subprocess,sys
from pathlib import Path
R=Path(__file__).parents[1]
def test_contract():
 subprocess.run([sys.executable,str(R/'experiments/live_cfg_chart_terminal_edge_repair_20260916.py')],check=True)
 d=json.loads((R/'runs/live-cfg-chart-terminal-edge-repair-20260916.json').read_text());x=d['rows'][0];assert d['summary']['max_length']>38;assert x['chart_state']['complete_clause_item'];assert x['audit']['sha256_equal'] is False;assert x['next_repair']
