import json
from pathlib import Path
def test_coordinated_place_lane():
 x=json.loads(Path('runs/cfg-locative-coordinated-place-20260917.json').read_text());assert x['control_count']==135168 and x['repair_count']==61440 and x['exact_count']==0
 for r in x['candidates']:
  assert ' and ' in r['rendered'];assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_preposition_coordination():
 assert 'preposition coordination' in json.loads(Path('runs/cfg-locative-coordinated-place-20260917.json').read_text())['next_repair']
