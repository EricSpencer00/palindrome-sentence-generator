import json
from pathlib import Path
def test_alternating_object_lane():
 x=json.loads(Path('runs/cfg-while-alternating-object-frame-20260917.json').read_text());assert x['candidate_count']>0 and x['pruned_partial_count']>0 and x['exact_count']==0
 for r in x['candidates']:
  assert r['grammar_state']['object_relation']=='alternating';assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_object_number():
 assert 'object-number agreement' in json.loads(Path('runs/cfg-while-alternating-object-frame-20260917.json').read_text())['next_repair']
