import json
from pathlib import Path
def test_causal_shared_subject_lane():
 x=json.loads(Path('runs/cfg-causal-shared-subject-agreement-20260917.json').read_text());assert x['candidate_count']>0 and x['pruned_partial_count']>0 and x['exact_count']==0
 for r in x['candidates']:
  assert r['novelty_preflight']['not_locative_family'];assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal'];assert r['grammar_state']['subject_sharing']
def test_next_past_state():
 assert 'past-tense agreement state' in json.loads(Path('runs/cfg-causal-shared-subject-agreement-20260917.json').read_text())['next_repair']
