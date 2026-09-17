import json
from pathlib import Path
def test_coordinated_subject_lane():
 x=json.loads(Path('runs/cfg-locative-coordinated-subject-20260917.json').read_text());assert x['control_count']==33792 and x['repair_count']==15360 and x['exact_count']==0
 for r in x['candidates']:
  assert ' and ' in r['rendered'];assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_locative_coordination():
 assert 'coordination inside the locative subject' in json.loads(Path('runs/cfg-locative-coordinated-subject-20260917.json').read_text())['next_repair']
