from experiments.abba_online_seam_generator_20260921 import run
def test_abba_online_lane():
 r=run(); assert r['novelty_preflight']['status']=='passed'; assert r['stats']['lattice_rows']==2; assert r['stats']['exact_count']==0; assert r['stats']['longest_letters']>38
 for x in r['candidates']: assert x['provenance']['distinct_surfaces'] and x['provenance']['online_character_obligations']; assert x['complete_prose']
def test_independent_audits():
 for x in run()['candidates']:
  assert x['audit']['sha256_forward'] != x['audit']['sha256_reverse']; assert not x['audit']['two_pointer_exact']; assert x['online_obligations'][0]['satisfied'] is False
