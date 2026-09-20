from unequal_center_buffer_grammar_20260920 import audit,run
def test_buffer_prunes_before_rendering():
 r=run(); assert r['stats']['frontier_pruned_before_rendering']>0; assert r['stats']['rendered_candidates']==0; assert r['stats']['fresh_exact_gt38']==0
def test_audit(): assert audit('A man, a plan, a canal: Panama!')['exact']
