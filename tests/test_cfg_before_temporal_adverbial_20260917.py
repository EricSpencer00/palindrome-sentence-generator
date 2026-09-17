import json
from pathlib import Path
def test_before_adverbial_lane():
 x=json.loads(Path('runs/cfg-before-temporal-adverbial-20260917.json').read_text());assert x['candidate_count']>0 and x['pruned_partial_count']>0 and x['exact_count']==0
 for r in x['candidates']:
  assert r['grammar_state']['adverbial_slot']=='post-before';assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal'];assert r['grammar_state']['tense_sequence']=='present->past'
def test_next_preconnective_slot():
 assert 'pre-connective adverbial slot' in json.loads(Path('runs/cfg-before-temporal-adverbial-20260917.json').read_text())['next_repair']
