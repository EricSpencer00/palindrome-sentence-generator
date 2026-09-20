from two_sided_unmatched_buffer_dp_20260920 import run
def test_actual_buffer_constructor_reaches_zero_frontier_without_rendering_invalid_states():
 x=run(); assert x['stats']['pruned_mismatch']>0; assert x['stats']['rendered_candidates']==0
 assert x['novelty_preflight']['status']=='zero-frontier'
