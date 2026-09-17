import json
from pathlib import Path
def test_joint_complementizer_lane():
 x=json.loads(Path('runs/cfg-joint-where-inwhich-alternation-20260917.json').read_text());assert x['control_count']==256 and x['repair_count']==256 and x['exact_count']==0
 for r in x['candidates']:
  assert r['complementizer'] in ('where','in which');assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_predicate_repair():
 assert 'joint locative predicate alternation' in json.loads(Path('runs/cfg-joint-where-inwhich-alternation-20260917.json').read_text())['next_repair']
