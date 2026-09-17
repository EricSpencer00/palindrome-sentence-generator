import json
from pathlib import Path
def test_second_plural_object_lane():
 x=json.loads(Path('runs/cfg-while-second-plural-object-substitution-20260917.json').read_text());assert x['candidate_count']>0 and x['pruned_partial_count']>0 and x['exact_count']==0
 for r in x['candidates']:
  z=r['grammar_state'];assert z['object_numbers']==('pl','pl') or z['object_numbers']==['pl','pl'];assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_paired_plural_objects():
 assert 'paired plural-object substitution' in json.loads(Path('runs/cfg-while-second-plural-object-substitution-20260917.json').read_text())['next_repair']
