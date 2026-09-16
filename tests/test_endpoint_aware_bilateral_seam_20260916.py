import json,subprocess,sys
from pathlib import Path
R=Path(__file__).parents[1]
def test_endpoint_seam_contract():
 subprocess.run([sys.executable,str(R/'experiments/endpoint_aware_bilateral_seam_20260916.py')],check=True)
 d=json.loads((R/'runs/endpoint-aware-bilateral-seam-20260916.json').read_text());assert d['novelty_preflight']['passed'];assert d['summary']['max_length']>100
 for x in d['rows']:
  assert x['seam_state']['endpoint_reservation_before_interior'] is True;assert x['audit']['sha256_equal'] is False;assert x['next_repair']
