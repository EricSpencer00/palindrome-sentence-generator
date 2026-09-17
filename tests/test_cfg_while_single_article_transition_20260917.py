import json
from pathlib import Path
def test_single_article_lane():
 x=json.loads(Path('runs/cfg-while-single-article-transition-20260917.json').read_text());assert x['candidate_count']>0 and x['pruned_partial_count']>0 and x['exact_count']==0
 for r in x['candidates']:
  assert r['grammar_state']['object_numbers'][0]!=r['grammar_state']['object_numbers'][1];assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_second_article():
 assert 'second object' in json.loads(Path('runs/cfg-while-single-article-transition-20260917.json').read_text())['next_repair']
