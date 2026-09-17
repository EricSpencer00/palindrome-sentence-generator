import json
from pathlib import Path
def test_matrix_determiner_lane():
 x=json.loads(Path('runs/cfg-shared-where-matrix-determiner-20260917.json').read_text());assert x['control_count']==256 and x['repair_count']==256 and x['exact_count']==0
 for r in x['candidates']:
  assert r['matrix_determiner'] in ('the','a');assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_plural_determiner():
 assert 'coordinated matrix determiner' in json.loads(Path('runs/cfg-shared-where-matrix-determiner-20260917.json').read_text())['next_repair']
