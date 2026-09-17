import json
from pathlib import Path
def test_object_number_lane():
 x=json.loads(Path('runs/cfg-while-alternating-object-number-20260917.json').read_text());assert x['candidate_count']>0 and x['pruned_partial_count']>0 and x['exact_count']==0
 for r in x['candidates']:
  assert r['grammar_state']['object_numbers'][0]!=r['grammar_state']['object_numbers'][1];assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_determiner_number():
 assert 'determiner-number' in json.loads(Path('runs/cfg-while-alternating-object-number-20260917.json').read_text())['next_repair']
