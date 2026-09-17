import json,subprocess,sys
from pathlib import Path
R=Path(__file__).parents[1]
def test_audit():
 subprocess.run([sys.executable,str(R/'experiments/audit_matrix_lanes_1_2_5_20260916.py')],check=True)
 d=json.loads((R/'runs/audit-matrix-lanes-1-2-5-20260916.json').read_text());assert d['search_executed'] is False;assert len(d['lanes'])==3
 for x in d['lanes']:
  assert x['provenance_present'] and x['novelty_preflight_present'] and x['next_repair_present']
  for a in x['independent_replay']: assert 'sha256_reverse' in a and 'two_pointer_exact' in a
