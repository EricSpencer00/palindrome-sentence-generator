import json
from pathlib import Path
def test_shared_where_agreement_lane():
 x=json.loads(Path('runs/cfg-shared-where-agreement-variation-20260917.json').read_text());assert x['control_count']==256 and x['repair_count']==256 and x['exact_count']==0
 for r in x['candidates']:
  assert r['agreement_state'] in ('singular','plural');assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_matrix_agreement():
 assert 'matrix-verb agreement' in json.loads(Path('runs/cfg-shared-where-agreement-variation-20260917.json').read_text())['next_repair']
