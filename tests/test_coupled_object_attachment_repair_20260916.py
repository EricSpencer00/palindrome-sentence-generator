from experiments.coupled_object_attachment_repair_20260916 import run
def test_coupled_repair_is_bounded_and_audited():
 p=run();assert p['stats']['joint_assignments']==4;assert p['novelty_preflight']['status']=='passed'
 for r in p['candidates']:
  assert r['letters']>100 and not r['exact_audit']['two_pointer_exact'];assert not r['exact_audit']['sha_equal'];assert r['provenance']['source_sentences_copied'] is False
