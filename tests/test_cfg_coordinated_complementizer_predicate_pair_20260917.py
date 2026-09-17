import json
from pathlib import Path
def test_coordinated_pair_lane():
 x=json.loads(Path('runs/cfg-coordinated-complementizer-predicate-pair-20260917.json').read_text());assert x['control_count']==256 and x['repair_count']==160 and x['exact_count']==0
 for r in x['candidates']:
  assert r['complementizer'] in ('where','in which');assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_determiner_pair():
 assert 'determiner/predicate-pair' in json.loads(Path('runs/cfg-coordinated-complementizer-predicate-pair-20260917.json').read_text())['next_repair']
