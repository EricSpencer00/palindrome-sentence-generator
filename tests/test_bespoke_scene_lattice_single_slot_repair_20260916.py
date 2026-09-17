import json,subprocess,sys
from pathlib import Path
R=Path(__file__).parents[1]
def test_contract():
 subprocess.run([sys.executable,str(R/'experiments/bespoke_scene_lattice_single_slot_repair_20260916.py')],check=True)
 d=json.loads((R/'runs/bespoke-scene-lattice-single-slot-repair-20260916.json').read_text());x=d['rows'][0];assert d['novelty_preflight']['passed'];assert x['repair_state']['single_state'];assert x['audit']['sha256_equal'] is False;assert x['next_repair']
