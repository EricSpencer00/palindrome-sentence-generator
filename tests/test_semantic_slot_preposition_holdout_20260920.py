from experiments import semantic_slot_preposition_holdout_20260920 as lane
def test_holdout_lane_audits_and_controls():
 r=lane.run(20); assert r['novelty_preflight']['status']=='passed'; assert r['stats']['states']==20; assert r['stats']['exact']==0
 assert {x['heldout_preposition'] for x in r['controls']}==set(lane.HOLDOUT)
 for x in r['controls']:
  assert x['audit']['sha256_forward']!=x['audit']['sha256_reverse']; assert x['provenance']['repair_after_render'] is False
