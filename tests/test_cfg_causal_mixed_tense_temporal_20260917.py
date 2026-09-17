import json
from pathlib import Path
def test_mixed_temporal_lane():
 x=json.loads(Path('runs/cfg-causal-mixed-tense-temporal-20260917.json').read_text());assert x['candidate_count']>0 and x['pruned_partial_count']>0 and x['exact_count']==0
 for r in x['candidates']:
  assert r['grammar_state']['tense_sequence']=='past->present';assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal'];assert r['grammar_state']['subject_sharing']
def test_next_before_direction():
 assert 'reverse present-to-past' in json.loads(Path('runs/cfg-causal-mixed-tense-temporal-20260917.json').read_text())['next_repair']
