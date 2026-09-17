import json
from pathlib import Path
def test_object_determiner_lane():
 x=json.loads(Path('runs/cfg-while-alternating-object-determiner-20260917.json').read_text());assert x['candidate_count']>0 and x['pruned_partial_count']>0 and x['exact_count']==0
 for r in x['candidates']:
  z=r['grammar_state'];assert z['object_numbers'][0]!=z['object_numbers'][1];assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_article_transition():
 assert 'article agreement transition' in json.loads(Path('runs/cfg-while-alternating-object-determiner-20260917.json').read_text())['next_repair']
