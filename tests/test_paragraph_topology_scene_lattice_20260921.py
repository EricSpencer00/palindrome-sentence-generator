from experiments.paragraph_topology_scene_lattice_20260921 import run
def test_bounded_topology():
 r=run(); assert r['novelty_preflight']['status']=='passed'; assert r['stats']['candidates']==4; assert r['stats']['exact_count']==0; assert r['stats']['longest_letters']>38
 for x in r['candidates']:
  assert x['complete_prose'] and x['provenance']['synchronous_generation'] and not x['provenance']['posthoc_repair']
  assert x['gates']['repeated_unit_absent'] and x['gates']['self_palindromic_unit_absent']
def test_audits():
 for x in run()['candidates']:
  assert x['audit']['sha256_forward'] != x['audit']['sha256_reverse']; assert not x['audit']['two_pointer_exact']
