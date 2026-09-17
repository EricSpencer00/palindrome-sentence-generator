import json
from pathlib import Path
def test_joint_predicate_lane():
 x=json.loads(Path('runs/cfg-joint-locative-predicate-alternation-20260917.json').read_text());assert x['control_count']==160 and x['repair_count']==96 and x['exact_count']==0
 for r in x['candidates']:
  assert r['attachment_state']=='joint-locative';assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_subject_repair():
 assert 'locative-subject lexical alternation' in json.loads(Path('runs/cfg-joint-locative-predicate-alternation-20260917.json').read_text())['next_repair']
