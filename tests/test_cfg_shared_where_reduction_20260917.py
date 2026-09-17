import json
from pathlib import Path
def test_shared_where_lane():
 x=json.loads(Path('runs/cfg-shared-where-reduction-20260917.json').read_text());assert x['control_count']==512 and x['repair_count']==512 and x['exact_count']==0
 for r in x['candidates']:
  assert r['complementizer'] in ('in which','where');assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_shared_predicate():
 assert 'shared-preposition lexical predicate substitution' in json.loads(Path('runs/cfg-shared-where-reduction-20260917.json').read_text())['next_repair']
