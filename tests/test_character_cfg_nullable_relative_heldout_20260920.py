from experiments.character_cfg_nullable_relative_heldout_20260920 import audit,run
def test_heldout_has_intact_and_shuffled_controls():
 r=run(); assert r['novelty_preflight']['status']=='passed'; assert r['reader_facing_candidates'] and r['shuffled_controls']; assert all(x['grammar']['nullable_pp'] and x['grammar']['subject_relative'] for x in r['reader_facing_candidates'])
def test_audit_exact():
 x=audit('A man, a plan, a canal: Panama.'); assert x['pointer_exact'] and x['sha256_forward']==x['sha256_reverse']
