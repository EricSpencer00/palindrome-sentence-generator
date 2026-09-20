from bidirectional_scene_lattice_20260920 import run
def test_scene_lattice_zero_frontier_and_controls():
 x=run(); assert x['stats']['rendered_controls']==3; assert x['stats']['live_closures']==0; assert x['stats']['exact_gt38']==0
 assert x['novelty_preflight']['catalogue_surface_reuse'] is False
