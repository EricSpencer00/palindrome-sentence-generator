import importlib.util
from pathlib import Path
p=Path(__file__).parents[1]; s=importlib.util.spec_from_file_location('x',p/'experiments/typed_template_equation_search_20260917.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
def test_seedless_bounded_branch_and_audits():
 x=m.run(); assert x['stats']['rendered']==81; assert x['stats']['longest_letters']>38; assert x['stats']['exact']==0
 for r in x['rendered_candidates'][:5]:
  assert r['audit']['sha256_forward']!=r['audit']['sha256_reverse']; assert r['audit']['two_pointer_exact'] is False
  assert r['trace'] and not r['proper_self_palindromic_spans']
def test_no_catalogue_or_shortcut_provenance():
 x=m.run(); assert x['provenance']['catalogue_used'] is False
 for r in x['rendered_candidates']: assert all(not r['provenance'][k] for k in ('catalogue_phrase_included','wrapped_seed','finished_tape_reversal','duplicate_bank_sweep'))
 assert 'typed lexical edge substitution' in x['next_repair']['operator']
