import json,subprocess,sys
from pathlib import Path
R=Path(__file__).parents[1]
def test_contract():
 subprocess.run([sys.executable,str(R/'experiments/bidirectional_phrase_pair_growth_20260916.py')],check=True)
 d=json.loads((R/'runs/bidirectional-phrase-pair-growth-20260916.json').read_text());assert d['summary']['max_length']>38;assert d['novelty_preflight']['passed']
 for x in d['rows']:
  assert x['audit']['sha256_equal'] is False;assert x['anti_shortcut']['fixed_tape'] is False;assert x['next_repair']
