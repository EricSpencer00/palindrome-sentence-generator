from experiments.residual_relation_setting_20260920 import audit,run
def test_two_slot_debt():
 r=run(); assert r['stats']['rendered_candidates']>0; assert r['stats']['fresh_exact_gt38']==0
 assert all(x['provenance']['debt_carried_across_two_slots'] for x in r['rendered_candidates'])
def test_audit(): assert audit('A man, a plan, a canal: Panama!')['exact']
