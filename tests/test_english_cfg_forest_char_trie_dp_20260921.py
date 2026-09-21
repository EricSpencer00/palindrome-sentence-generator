from experiments.english_cfg_forest_char_trie_dp_20260921 import audit,run
def test_forest_is_broad_and_live():
 r=run(); assert r['stats']['independent_pairs']>38 and r['stats']['live_pruned']; assert r['rendered_controls']
def test_audit():
 a=audit('A man, a plan, a canal, panama.'); assert a['pointer_exact'] and a['sha256_forward']==a['sha256_reverse']
