import importlib.util
from pathlib import Path
p=Path(__file__).parents[1]; s=importlib.util.spec_from_file_location('x',p/'experiments/seam_indexed_typed_repair_20260917.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_bounded_seam_repair_audited():
 x=m.run(); assert x['stats']['rendered']==6 and x['stats']['longest_letters']>38
 for r in x['rendered_candidates']:
  assert r['provenance']['seam_indexed'] and r['before_residual']['debt']>=0
  assert r['audit']['sha256_forward']!=r['audit']['sha256_reverse']
def test_anti_shortcut_and_next_repair():
 x=m.run(); assert x['stats']['exact']==0
 assert not x['provenance']['catalogue_used'] and not x['provenance']['finished_tape_reversal']
 assert 'joint two-slot' in x['next_repair']['operator']
