import json
from pathlib import Path
def test_formerly_later_lane():
 x=json.loads(Path('runs/cfg-before-formerly-later-order-20260917.json').read_text());assert x['candidate_count']>0 and x['pruned_partial_count']>0 and x['exact_count']==0
 for r in x['candidates']:
  assert r['grammar_state']['temporal_order']=='formerly<later';assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_while_variant():
 assert 'while' in json.loads(Path('runs/cfg-before-formerly-later-order-20260917.json').read_text())['next_repair']
