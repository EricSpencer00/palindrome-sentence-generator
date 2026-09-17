import json
from pathlib import Path
def test_matrix_agreement_lane():
 x=json.loads(Path('runs/cfg-shared-where-matrix-agreement-20260917.json').read_text());assert x['control_count']==256 and x['repair_count']==256 and x['exact_count']==0
 for r in x['candidates']:
  assert r['matrix_agreement'] in ('singular-matrix-subject','plural-coordinated-matrix-subject');assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_matrix_subject_repair():
 assert 'matrix determiner' in json.loads(Path('runs/cfg-shared-where-matrix-agreement-20260917.json').read_text())['next_repair']
