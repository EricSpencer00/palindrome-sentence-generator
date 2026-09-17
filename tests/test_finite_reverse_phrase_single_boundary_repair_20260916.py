import json,subprocess,sys
from pathlib import Path
R=Path(__file__).parents[1]
def test_contract():
 subprocess.run([sys.executable,str(R/'experiments/finite_reverse_phrase_single_boundary_repair_20260916.py')],check=True)
 d=json.loads((R/'runs/finite-reverse-phrase-single-boundary-repair-20260916.json').read_text());x=d['rows'][0];assert x['repair_state']['single_state'];assert x['audit']['sha256_equal'] is False;assert x['next_repair']
