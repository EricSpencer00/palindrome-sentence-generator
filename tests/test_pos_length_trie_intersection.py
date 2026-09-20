from pos_length_trie_intersection_20260920 import run
def test_pos_trie_frontier():
 x=run(); assert x['stats']['states']==9; assert x['stats']['live_closed']==0; assert x['stats']['exact_gt38']==0
 assert all(r['provenance']['reverse_trie_intersection'] for r in x['candidates'])
