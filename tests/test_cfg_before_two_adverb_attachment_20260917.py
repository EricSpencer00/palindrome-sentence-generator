import json
from pathlib import Path
def test_two_adverb_lane():
 x=json.loads(Path('runs/cfg-before-two-adverb-attachment-20260917.json').read_text());assert x['candidate_count']>0 and x['pruned_partial_count']>0 and x['exact_count']==0
 for r in x['candidates']:
  z=r['grammar_state']['adverbial_slots'];assert z['pre_event']!=z['post_event'];assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_slot_repair():
 assert 'one attachment slot at a time' in json.loads(Path('runs/cfg-before-two-adverb-attachment-20260917.json').read_text())['next_repair']
