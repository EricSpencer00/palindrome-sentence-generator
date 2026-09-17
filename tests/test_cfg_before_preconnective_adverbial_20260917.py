import json
from pathlib import Path
def test_preconnective_lane():
 x=json.loads(Path('runs/cfg-before-preconnective-adverbial-20260917.json').read_text());assert x['candidate_count']>0 and x['pruned_partial_count']>0 and x['exact_count']==0
 for r in x['candidates']:
  assert r['grammar_state']['adverbial_slot']=='pre-connective';assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_two_adverbs():
 assert 'two-adverb temporal branch' in json.loads(Path('runs/cfg-before-preconnective-adverbial-20260917.json').read_text())['next_repair']
