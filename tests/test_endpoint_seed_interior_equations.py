from endpoint_seed_interior_equations_20260920 import run
def test_endpoint_seed_does_not_claim_construction():
 x=run(); assert x['stats']['endpoint_seed_matches']==9
 assert x['stats']['interior_live_closures']==0 and x['stats']['exact_candidates']==0
