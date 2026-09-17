import json
from pathlib import Path
def test_locative_subject_lane():
 x=json.loads(Path('runs/cfg-inwhich-locative-subject-20260917.json').read_text());assert x['control_count']==9216 and x['repair_count']==3072 and x['exact_count']==0
 for r in x['candidates']:
  assert r['attachment_state']=='locative-in-which';assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_joint_repair():
 assert 'matrix subject substitution jointly' in json.loads(Path('runs/cfg-inwhich-locative-subject-20260917.json').read_text())['next_repair']
