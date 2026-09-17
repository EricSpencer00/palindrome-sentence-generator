from experiments.centerout_museum_scene_lattice_20260916 import run
def test_centerout_museum_lattice_is_fresh_and_audited():
 p=run();assert p['stats']['joint_states']==4;assert p['novelty_preflight']['status']=='passed';assert not p['novelty_preflight']['old_courier_harbor_frame_used']
 for r in p['candidates']:
  assert r['letters']>38;assert not r['exact_audit']['two_pointer_exact'];assert not r['exact_audit']['sha_equal'];assert r['provenance']['agreement_and_valency_checked']
