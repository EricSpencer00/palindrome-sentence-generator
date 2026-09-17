import json
from pathlib import Path
def test_coordinated_subject_lexical_lane():
 x=json.loads(Path('runs/cfg-shared-where-coordinated-subject-lexical-20260917.json').read_text());assert x['control_count']==256 and x['repair_count']==256 and x['exact_count']==0
 for r in x['candidates']:
  assert ' and ' in r['rendered'];assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_joint_subject_repair():
 assert 'locative subject lexical substitution jointly' in json.loads(Path('runs/cfg-shared-where-coordinated-subject-lexical-20260917.json').read_text())['next_repair']
