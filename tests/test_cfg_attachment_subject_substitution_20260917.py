import json
from pathlib import Path
def test_subject_substitution_lane():
 x=json.loads(Path('runs/cfg-attachment-subject-substitution-20260917.json').read_text());assert x['control_count']==768 and x['repair_count']==256 and x['exact_count']==0
 for r in x['candidates']:
  assert r['novelty_preflight']['valency_preserved'];assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_matrix_object_repair():
 assert 'matrix object substitution' in json.loads(Path('runs/cfg-attachment-subject-substitution-20260917.json').read_text())['next_repair']
