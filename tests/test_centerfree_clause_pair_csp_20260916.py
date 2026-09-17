from experiments.centerfree_clause_pair_csp_20260916 import run
def test_centerfree_pair_has_live_seam_and_no_fixed_center():
 p=run();assert p['stats']['joint_states']==4;assert p['novelty_preflight']['status']=='passed';assert not p['novelty_preflight']['fixed_center_or_tape']
 for r in p['candidates']:
  assert r['letters']>38;assert not r['exact_audit']['two_pointer_exact'];assert not r['exact_audit']['sha_equal'];assert r['provenance']['old_scene_family_reused'] is False
