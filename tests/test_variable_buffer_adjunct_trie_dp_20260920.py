from variable_buffer_adjunct_trie_dp_20260920 import run
def test_variable_buffer_lane_prunes_before_rendering():
 x=run(); assert x['stats']['pruned_mismatch']>0; assert x['stats']['rendered_candidates']==0; assert x['novelty_preflight']['status']=='zero-frontier'
