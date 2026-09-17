from experiments.bilateral_semantic_growth_grammar_20260916 import run
def test_growth_states_are_fresh_prose_and_independently_audited():
 p=run();assert p['stats']['states']==3;assert p['novelty_preflight']['status']=='passed'
 assert p['stats']['largest_letters']>100
 for r in p['growth_states']:
  assert r['exact_audit']['algorithm'].startswith('independent_two_pointer');assert not r['exact_audit']['two_pointer_exact'];assert not r['exact_audit']['sha_equal'];assert r['provenance']['source_sentences_copied'] is False
