import json
from pathlib import Path
def test_inwhich_preposition_lane():
 x=json.loads(Path('runs/cfg-inwhich-location-preposition-20260917.json').read_text());assert x['control_count']==2048 and x['repair_count']==1024 and x['exact_count']==0
 for r in x['candidates']:
  assert r['attachment_state']=='locative-in-which';assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_locative_predicate():
 assert 'locative predicate substitution' in json.loads(Path('runs/cfg-inwhich-location-preposition-20260917.json').read_text())['next_repair']
