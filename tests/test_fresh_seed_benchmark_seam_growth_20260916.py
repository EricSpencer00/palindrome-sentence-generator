from experiments.fresh_seed_benchmark_seam_growth_20260916 import run
def test_seed_is_benchmark_only_and_seam_growth_is_audited():
 p=run();assert p['benchmark']['used_as_output'] is False;assert p['stats']['heldout_assignments']==16;assert p['novelty_preflight']['status']=='passed'
 for r in p['candidates']:
  assert r['letters']>38;assert not r['exact_audit']['two_pointer_exact'];assert not r['exact_audit']['sha_equal'];assert r['provenance']['seed_wrapped'] is False
