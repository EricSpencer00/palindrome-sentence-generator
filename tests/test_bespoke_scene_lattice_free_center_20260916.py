import json,subprocess,sys
from pathlib import Path
R=Path(__file__).parents[1]
def test_contract():
 subprocess.run([sys.executable,str(R/'experiments/bespoke_scene_lattice_free_center_20260916.py')],check=True)
 d=json.loads((R/'runs/bespoke-scene-lattice-free-center-20260916.json').read_text());assert d['novelty_preflight']['passed'];assert d['summary']['candidate_count']==8;assert d['summary']['max_length']>38
 for x in d['rows']:
  assert x['audit']['sha256_equal'] is False;assert x['lattice_state']['free_center']=='semicolon';assert x['next_repair']
