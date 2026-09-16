from experiments.exact_candidate_slot_repair_neighborhood_20260916 import run
def test_typed_repair_neighborhood_is_small_and_audited():
 p=run();assert p['stats']['bounded_substitutions']==6;assert p['stats']['exact']==0
 assert p['novelty_preflight']['status']=='passed'
 for r in p['candidates']:
  assert r['letters']>=90 and not r['exact_audit']['two_pointer_exact']
  assert r['exact_audit']['sha_equal'] is False
  assert r['provenance']['source_sentences_copied'] is False
 assert p['follow_up']['follow_up_repair']['applied'] is False
