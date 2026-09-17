import json
from pathlib import Path
def test_joint_locative_subject_lane():
 x=json.loads(Path('runs/cfg-shared-where-joint-locative-subject-20260917.json').read_text());assert x['control_count']==192 and x['repair_count']==256 and x['exact_count']==0
 for r in x['candidates']:
  assert ' and ' in r['rendered'];assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_place_pair_repair():
 assert 'joint place-pair substitution' in json.loads(Path('runs/cfg-shared-where-joint-locative-subject-20260917.json').read_text())['next_repair']
