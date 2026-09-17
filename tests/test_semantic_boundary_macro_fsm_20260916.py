from experiments.semantic_boundary_macro_fsm_20260916 import run
def test_macro_fsm_emits_long_prose_without_mirrored_word_order():
 p=run();assert p['stats']['states']==3;assert p['stats']['largest_letters']>50;assert p['novelty_preflight']['status']=='passed'
 for r in p['candidates']:
  assert not r['exact_audit']['two_pointer_exact'];assert not r['exact_audit']['sha_equal'];assert r['provenance']['repeated_self_palindromic_unit'] is False
