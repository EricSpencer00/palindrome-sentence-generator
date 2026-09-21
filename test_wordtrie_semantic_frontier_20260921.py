from experiments.wordtrie_semantic_frontier_20260921 import run
def test_semantic_trie_reports_frontier_and_audit():
 r=run(); assert r['novelty_preflight']['status']=='passed'; assert r['stats']['exact_gt38']==len(r['exact_candidates']); assert r['deepest_grammar_frontier']['next_lexicon_change']
