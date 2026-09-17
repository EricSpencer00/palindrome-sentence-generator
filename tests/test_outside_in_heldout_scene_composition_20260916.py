import json,subprocess,sys
from pathlib import Path
R=Path(__file__).parents[1]
def test_contract():
 subprocess.run([sys.executable,str(R/'experiments/outside_in_heldout_scene_composition_20260916.py')],check=True)
 d=json.loads((R/'runs/outside-in-heldout-scene-20260916.json').read_text());assert d['novelty_preflight']['passed'];assert d['summary']['max_length']>38
 for x in d['rows']:
  assert x['audit']['sha256_forward_normalized']!=x['audit']['sha256_reverse_normalized'];assert x['anti_shortcut']['posthoc_reversal'] is False;assert x['next_repair']
