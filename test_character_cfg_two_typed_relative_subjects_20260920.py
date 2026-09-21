from character_cfg_two_typed_relative_subjects_20260920 import audit,run
def test_typed_relatives_and_controls():
 r=run(); assert r['novelty_preflight']['status']=='passed'; assert r['reader_facing_candidates'] and r['shuffled_controls']; assert all(len(x['grammar']['relative_subjects'])==2 for x in r['reader_facing_candidates'])
def test_hash_audit():
 x=audit('A man, a plan, a canal: Panama.'); assert x['pointer_exact'] and x['sha256_forward']==x['sha256_reverse']
