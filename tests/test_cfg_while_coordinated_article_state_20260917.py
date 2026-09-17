import json
from pathlib import Path
def test_coordinated_article_lane():
 x=json.loads(Path('runs/cfg-while-coordinated-article-state-20260917.json').read_text());assert x['candidate_count']>0 and x['pruned_partial_count']>0 and x['exact_count']==0
 for r in x['candidates']:
  z=r['grammar_state'];assert z['coordinated_determiners'][0]==z['coordinated_determiners'][1];assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_plural_gating():
 assert 'plural-object' in json.loads(Path('runs/cfg-while-coordinated-article-state-20260917.json').read_text())['next_repair']
