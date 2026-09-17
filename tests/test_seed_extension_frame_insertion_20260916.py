from experiments.seed_extension_frame_insertion_20260916 import run
def test_frame_probe_forbids_seed_wrapper_and_keeps_prose():
 p=run();assert p['seed_policy']['wrapper_rejected'];assert p['stats']['frame_center_assignments']==6
 assert p['novelty_preflight']['status']=='passed'
 for r in p['candidates']:
  assert r['letters']>100 and not r['exact_audit']['two_pointer_exact'];assert not r['exact_audit']['sha_equal'];assert r['provenance']['historical_seed_used_as_output'] is False
