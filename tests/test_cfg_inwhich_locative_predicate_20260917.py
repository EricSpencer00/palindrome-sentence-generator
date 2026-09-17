import json
from pathlib import Path
def test_locative_predicate_lane():
 x=json.loads(Path('runs/cfg-inwhich-locative-predicate-20260917.json').read_text());assert x['control_count']==2304 and x['repair_count']==768 and x['exact_count']==0
 for r in x['candidates']:
  assert r['attachment_state']=='locative-in-which';assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_locative_subject():
 assert 'locative subject substitution' in json.loads(Path('runs/cfg-inwhich-locative-predicate-20260917.json').read_text())['next_repair']
