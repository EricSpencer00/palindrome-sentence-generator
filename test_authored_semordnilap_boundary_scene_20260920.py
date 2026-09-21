from authored_semordnilap_boundary_scene_20260920 import run
def test_probe():
 r=run(); assert r['novelty_preflight']['status']=='passed'; assert r['stats']['products']==32; assert r['best_control']; assert r['stats']['exact_clean']==0; assert all(not x['reader_eligible'] for x in r['candidates'])
def test_audits():
 for x in run()['candidates']:
  assert x['provenance']['construction_vocabulary_only']; assert x['complete_prose']; assert x['audit']['sha256_forward']; assert not x['provenance']['finished_tape_reversal']
