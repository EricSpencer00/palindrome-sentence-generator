import json
from pathlib import Path
def test_order_feature_lane():
 x=json.loads(Path('runs/cfg-before-earlier-later-order-20260917.json').read_text());assert x['candidate_count']>0 and x['pruned_partial_count']>0 and x['exact_count']==0
 for r in x['candidates']:
  assert r['grammar_state']['temporal_order']=='earlier<later';assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_formerly_later():
 assert 'formerly/later' in json.loads(Path('runs/cfg-before-earlier-later-order-20260917.json').read_text())['next_repair']
