import json
from pathlib import Path
def test_midpoint_product_lane():
 x=json.loads(Path('runs/cfg-midpoint-role-automaton-intersection-20260917.json').read_text());assert x['candidate_count']==256 and x['exact_count']==0
 for r in x['diagnostic_controls']:
  assert r['novelty_preflight']['complete_obligation_required'];assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal'];assert r['grammar_state']['role_state']['distinct_roles']
def test_no_near_miss_admission():
 x=json.loads(Path('runs/cfg-midpoint-role-automaton-intersection-20260917.json').read_text());assert x['admitted_renderings']==[]
