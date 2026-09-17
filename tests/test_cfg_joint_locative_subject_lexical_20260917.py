import json
from pathlib import Path
def test_joint_locative_subject_lexical_lane():
 x=json.loads(Path('runs/cfg-joint-locative-subject-lexical-20260917.json').read_text());assert x['control_count']==256 and x['repair_count']==256 and x['exact_count']==0
 for r in x['candidates']:
  assert ' and ' in r['rendered'];assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_place_pair_lexical():
 assert 'place-pair lexical alternation' in json.loads(Path('runs/cfg-joint-locative-subject-lexical-20260917.json').read_text())['next_repair']
