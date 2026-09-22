from experiments.compositional_slot_carry_20260920 import audit,run
def test_slot_carry_lane():
 r=run(); assert r['novelty_preflight']['status']=='passed'; assert r['stats']['slot_states']>0; assert r['stats']['fresh_exact_gt38']==0
 for x in r['rendered_candidates']:
  assert x['provenance']['constraint_carried_between_slots']; assert len(x['audit']['sha256_forward'])==64
def test_audit(): assert audit('A man, a plan, a canal: Panama!')['exact']
