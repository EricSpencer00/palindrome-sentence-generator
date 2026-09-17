import json
from pathlib import Path
def test_causal_past_lane():
 x=json.loads(Path('runs/cfg-causal-past-tense-agreement-20260917.json').read_text());assert x['candidate_count']>0 and x['pruned_partial_count']>0 and x['exact_count']==0
 for r in x['candidates']:
  assert r['grammar_state']['tense']=='past';assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal'];assert r['grammar_state']['subject_sharing']
def test_next_mixed_tense_state():
 assert 'mixed-tense causal state' in json.loads(Path('runs/cfg-causal-past-tense-agreement-20260917.json').read_text())['next_repair']
