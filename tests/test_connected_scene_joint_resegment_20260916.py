from experiments.connected_scene_joint_resegment_20260916 import run
def test_joint_scene_resegment_is_bounded_and_exactly_audited():
 p=run();assert p['stats']['joint_assignments']==4;assert p['novelty_preflight']['status']=='passed'
 for r in p['candidates']:
  assert r['letters']>38;assert not r['exact_audit']['two_pointer_exact'];assert not r['exact_audit']['sha_equal'];assert r['provenance']['connected_scene']
