import json
from pathlib import Path
def test_locative_pronoun_lane():
 x=json.loads(Path('runs/cfg-locative-where-inwhich-20260917.json').read_text());assert x['control_count']==3072 and x['repair_count']==3072 and x['exact_count']==0
 for r in x['candidates']:
  assert r['attachment_state']=='locative';assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_location_preposition():
 assert 'location-preposition substitution' in json.loads(Path('runs/cfg-locative-where-inwhich-20260917.json').read_text())['next_repair']
