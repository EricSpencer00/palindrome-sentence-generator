import json
from pathlib import Path
def test_joint_subject_lane():
 x=json.loads(Path('runs/cfg-joint-matrix-locative-subject-20260917.json').read_text());assert x['control_count']==10752 and x['repair_count']==1536 and x['exact_count']==0
 for r in x['candidates']:
  assert r['novelty_preflight']['dual_valency_preserved'];assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_coordination_repair():
 assert 'coordinated-subject realization' in json.loads(Path('runs/cfg-joint-matrix-locative-subject-20260917.json').read_text())['next_repair']
