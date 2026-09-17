import json
from pathlib import Path
def test_post_slot_lane():
 x=json.loads(Path('runs/cfg-before-post-slot-adverb-substitution-20260917.json').read_text());assert x['candidate_count']>0 and x['pruned_partial_count']>0 and x['exact_count']==0
 for r in x['candidates']:
  assert r['grammar_state']['tense_sequence']=='present->past';assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_pair_state():
 assert 'paired adverb lexical agreement' in json.loads(Path('runs/cfg-before-post-slot-adverb-substitution-20260917.json').read_text())['next_repair']
