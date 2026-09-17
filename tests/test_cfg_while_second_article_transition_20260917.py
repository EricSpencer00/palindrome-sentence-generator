import json
from pathlib import Path
def test_second_article_lane():
 x=json.loads(Path('runs/cfg-while-second-article-transition-20260917.json').read_text());assert x['candidate_count']>0 and x['pruned_partial_count']>0 and x['exact_count']==0
 for r in x['candidates']:
  assert r['grammar_state']['article_transition'] in ('none','second_object_article');assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_coordinated_articles():
 assert 'coordinated article state' in json.loads(Path('runs/cfg-while-second-article-transition-20260917.json').read_text())['next_repair']
