import json
from pathlib import Path
def test_new_discourse_cfg_lane():
 x=json.loads(Path('runs/cfg-discourse-relation-obligation-intersection-20260917.json').read_text());assert x['candidate_count']>0 and x['pruned_partial_count']>0 and x['exact_count']==0
 for r in x['candidates']:
  assert r['novelty_preflight']['not_prior_locative_family'];assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal'];assert any(q in r['rendered'] for q in ('because','although','while'))
def test_next_is_new_family():
 assert 'do not re-enter the locative family' in json.loads(Path('runs/cfg-discourse-relation-obligation-intersection-20260917.json').read_text())['next_repair']
