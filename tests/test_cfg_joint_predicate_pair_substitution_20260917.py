import json
from pathlib import Path
def test_predicate_pair_lane():
 x=json.loads(Path('runs/cfg-joint-predicate-pair-substitution-20260917.json').read_text());assert x['control_count']==256 and x['repair_count']==192 and x['exact_count']==0
 for r in x['candidates']:
  assert '/' in r['predicate_pair'];assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_complementizer_pair():
 assert 'complementizer/predicate pair' in json.loads(Path('runs/cfg-joint-predicate-pair-substitution-20260917.json').read_text())['next_repair']
