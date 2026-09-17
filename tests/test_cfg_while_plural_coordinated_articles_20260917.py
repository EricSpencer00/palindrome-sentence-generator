import json
from pathlib import Path
def test_plural_articles_lane():
 x=json.loads(Path('runs/cfg-while-plural-coordinated-articles-20260917.json').read_text());assert x['candidate_count']>0 and x['pruned_partial_count']>0 and x['exact_count']==0
 for r in x['candidates']:
  z=r['grammar_state'];assert z['object_numbers']==['pl','pl'] or tuple(z['object_numbers'])==('pl','pl');assert z['coordinated_determiners']==['the','the'] or tuple(z['coordinated_determiners'])==('the','the');assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_plural_substitution():
 assert 'plural-object lexical substitution' in json.loads(Path('runs/cfg-while-plural-coordinated-articles-20260917.json').read_text())['next_repair']
