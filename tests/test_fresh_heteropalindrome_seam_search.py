from fresh_heteropalindrome_seam_search_20260920 import run
def test_fresh_seam_controls_and_heldout_anchor():
 x=run(); assert x['stats']['rendered_candidates']==16; assert x['stats']['exact_gt38']==0
 assert x['held_out_anchor']['emitted'] is False
 assert all(r['provenance']['pointer_audit'] for r in x['best_controls'])
