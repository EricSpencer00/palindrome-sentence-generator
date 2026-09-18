import importlib.util
from pathlib import Path
p=Path(__file__).parents[1];s=importlib.util.spec_from_file_location('x',p/'experiments/two_sided_constituent_agreement_20260917.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_two_sided_constituent_repair():
 x=m.run(); assert x['stats']['rendered']==4 and x['stats']['longest_letters']>38
 for r in x['rendered_candidates']:
  assert r['agreement']['checked'] and r['provenance']['outer_assignments_mutated'] is False
  assert r['audit']['sha256_forward']!=r['audit']['sha256_reverse']
def test_provenance_and_next_repair():
 x=m.run(); assert x['stats']['exact']==0 and x['provenance']['catalogue_used'] is False
 assert 'object spans' in x['next_repair']['operator']
