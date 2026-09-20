from manual_outer8_inward_expansion_20260920 import run
def test_outer_equation_frontier():
 x=run(); assert x['stats']['pairs']==2; assert x['stats']['outer8_seeds']==0; assert x['stats']['exact_gt38']==0
 assert x['best_control']['audit']['letters']>38
