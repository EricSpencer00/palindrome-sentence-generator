import json
from pathlib import Path
def test_shared_where_predicate_lane():
 x=json.loads(Path('runs/cfg-shared-where-predicate-substitution-20260917.json').read_text());assert x['control_count']==368 and x['repair_count']==144 and x['exact_count']==0
 for r in x['candidates']:
  assert r['attachment_state']=='locative-where-shared-preposition';assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_agreement_repair():
 assert 'agreement variant' in json.loads(Path('runs/cfg-shared-where-predicate-substitution-20260917.json').read_text())['next_repair']
