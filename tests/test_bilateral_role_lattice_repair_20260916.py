from experiments.bilateral_role_lattice_repair_20260916 import run
def test_bilateral_role_repair_is_joint_and_audited():
 p=run();assert p['stats']['joint_assignments']==4;assert p['novelty_preflight']['status']=='passed'
 for r in p['candidates']:
  assert r['letters']>38;assert not r['exact_audit']['two_pointer_exact'];assert not r['exact_audit']['sha_equal'];assert r['provenance']['semordnilap_chain'] is False
