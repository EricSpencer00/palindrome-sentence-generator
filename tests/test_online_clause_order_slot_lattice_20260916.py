from experiments.online_clause_order_slot_lattice_20260916 import run
def test_online_lattice_emits_prose_and_independent_failure_audits():
 p=run(); assert p['stats']['orders']==6; assert p['stats']['exact']==0
 assert p['novelty_preflight']['status']=='passed'
 for r in p['candidates']:
  assert r['letters']>100 and r['exact_audit']['two_pointer_exact'] is False
  assert r['exact_audit']['sha_equal'] is False
  assert r['provenance']['source_sentences_copied'] is False
