import json,subprocess,sys
from pathlib import Path
R=Path(__file__).parents[1]
def test_contract():
 subprocess.run([sys.executable,str(R/'experiments/live_cfg_character_chart_20260916.py')],check=True)
 d=json.loads((R/'runs/live-cfg-character-chart-20260916.json').read_text());assert d['novelty_preflight']['passed'];assert d['summary']['max_length']>38
 for x in d['rows']:
  assert x['chart_state']['character_intersection'];assert x['audit']['sha256_equal'] is False;assert x['anti_shortcut']['disconnected_semordnilap_chain'] is False;assert x['next_repair']
