from experiments.live_boundary_shift_grammar_20260920 import run
def test_live_frontier_is_precise_zero():
 x=run(); assert x['stats']['transitions']==9
 assert x['stats']['boundary_closed']==0
 assert x['stats']['exact_closed']==0
 assert x['status'].startswith('precise zero frontier')
 assert all(not r['provenance']['post_hoc_repair'] for r in x['frontier'])
