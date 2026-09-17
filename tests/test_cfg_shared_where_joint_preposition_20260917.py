import json
from pathlib import Path
def test_joint_preposition_lane():
 x=json.loads(Path('runs/cfg-shared-where-joint-preposition-20260917.json').read_text());assert x['control_count']==176 and x['repair_count']==96 and x['exact_count']==0
 for r in x['candidates']:
  assert r['shared_preposition'] in ('near','beside','under');assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_complementizer():
 assert 'complementizer alternation' in json.loads(Path('runs/cfg-shared-where-joint-preposition-20260917.json').read_text())['next_repair']
