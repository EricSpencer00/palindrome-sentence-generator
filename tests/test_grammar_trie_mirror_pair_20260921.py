from experiments.grammar_trie_mirror_pair_20260921 import run
def test_trie_lane_records_deepest_original_parse():
 r=run(); assert r['stats']['exact_pairs']==0; assert r['stats']['deepest_prefix']>=0; assert r['deepest_parse']['provenance']['catalogue'] is False
