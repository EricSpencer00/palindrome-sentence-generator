import json
from pathlib import Path
import importlib.util
_spec=importlib.util.spec_from_file_location("residual", Path(__file__).parents[1]/"experiments/residual_scene_operator_20260917.py")
_mod=importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
run, letters, obligation = _mod.run, _mod.letters, _mod.obligation

def test_rendered_candidates_have_live_residual_trace_and_audit():
 p=run(); assert p['stats']['rendered']==2; assert p['stats']['longest_letters']>60
 for c in p['candidates']:
  assert c['rendered'].strip().endswith('.')
  assert c['audit']['two_pointer_exact'] is False
  assert c['audit']['sha256_forward'] != c['audit']['sha256_reverse']
  assert len(c['trace'])==4 and all('residual' in x for x in c['trace'])

def test_operator_constraints_and_no_shortcuts():
 p=run()
 for c in p['candidates']:
  assert c['provenance']['centre_nonpalindromic']
  assert c['provenance']['character_obligation_used']
  assert not c['provenance']['reversed_finished_sentence']
  assert not c['provenance']['duplicate_bank_sweep']
 assert p['next_repair']['operator'].startswith('add a boundary-aware')

def test_obligation_is_character_level_not_word_level():
 assert obligation('marks the tide','at dawn')['next_left'] == 'm'
 assert obligation('ab','ba')['closed'] is True
