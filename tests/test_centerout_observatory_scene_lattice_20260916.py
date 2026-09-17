from experiments.centerout_observatory_scene_lattice_20260916 import run
def test_observatory_centerout_lattice_is_bounded_and_non_nested():
 p=run();assert p['stats']['bounded_states']==4;assert p['novelty_preflight']['status']=='passed';assert p['novelty_preflight']['single_bounded_run']
 for r in p['candidates']:
  assert r['letters']>38;assert not r['exact_audit']['two_pointer_exact'];assert not r['exact_audit']['sha_equal'];assert r['provenance']['nested_palindrome_span'] is False
