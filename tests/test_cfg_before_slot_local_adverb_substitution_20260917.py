import json
from pathlib import Path
def test_slot_local_lane():
 x=json.loads(Path('runs/cfg-before-slot-local-adverb-substitution-20260917.json').read_text());assert x['candidate_count']>0 and x['pruned_partial_count']>0 and x['exact_count']==0
 for r in x['candidates']:
  assert r['grammar_state']['tense_sequence']=='present->past';assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_post_slot():
 assert 'post-event slot substitution' in json.loads(Path('runs/cfg-before-slot-local-adverb-substitution-20260917.json').read_text())['next_repair']
