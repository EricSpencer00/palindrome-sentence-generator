import json,subprocess,sys
from pathlib import Path
R=Path(__file__).parents[1]
def test_contract():
 subprocess.run([sys.executable,str(R/'experiments/centerout_open_word_seam_20260916.py')],check=True)
 d=json.loads((R/'runs/centerout-open-word-seam-20260916.json').read_text());assert d['novelty_preflight']['passed'];assert d['summary']['max_length']>100
 for x in d['rows']:
  assert x['seam_trace']['live_boundary_obligation'];assert x['audit']['sha256_equal'] is False;assert x['anti_shortcut']['dangling_word_reversal'] is False;assert x['next_repair']
