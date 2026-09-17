import json,subprocess,sys
from pathlib import Path
R=Path(__file__).parents[1]
def test_dp_contract():
 subprocess.run([sys.executable,str(R/'experiments/compositional_slot_boundary_dp_20260916.py')],check=True)
 d=json.loads((R/'runs/compositional-slot-boundary-dp-20260916.json').read_text());assert d['summary']['max_length']>100;assert d['novelty_preflight']['passed']
 for x in d['rows']:
  assert x['dp_trace']['nested_spans'] is False;assert x['audit']['sha256_equal'] is False;assert x['audit']['independent_two_pointer_exact'] is False;assert x['next_repair']
