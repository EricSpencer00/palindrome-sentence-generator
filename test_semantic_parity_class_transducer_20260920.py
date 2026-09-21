from semantic_parity_class_transducer_20260920 import run
def test_bounded_equivalence_transducer():
 r=run(); assert r['novelty_preflight']['status']=='passed'; assert r['stats']['equivalence_classes']==3; assert r['stats']['rendered_candidates']>0; assert r['stats']['live_prunes']>0
 assert all(not x['reader_eligible'] for x in r['candidates'])
def test_independent_audits_and_exclusions():
 for x in run()['candidates']:
  assert x['provenance']['fresh_independent_sides'] and x['provenance']['online_frontier']; assert not x['provenance']['finished_tape_reversal']; assert x['audit']['sha256_forward']
