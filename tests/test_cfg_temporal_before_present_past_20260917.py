import json
from pathlib import Path
def test_before_reverse_lane():
 x=json.loads(Path('runs/cfg-temporal-before-present-past-20260917.json').read_text());assert x['candidate_count']>0 and x['pruned_partial_count']>0 and x['exact_count']==0
 for r in x['candidates']:
  assert r['grammar_state']['tense_sequence']=='present->past';assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal'];assert ' before ' in r['rendered']
def test_next_adverbial_repair():
 assert 'temporal adverbial insertion' in json.loads(Path('runs/cfg-temporal-before-present-past-20260917.json').read_text())['next_repair']
