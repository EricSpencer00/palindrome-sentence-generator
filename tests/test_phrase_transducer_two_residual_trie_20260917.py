from experiments.phrase_transducer_two_residual_trie_20260917 import run
def test_two_residual_trie():
 x=run();assert x['candidate_count']==8 and x['exact_count']==0 and x['stats']['two_residual_branches']==2
 assert all(r['live_obligations']['rendered_after_constraints'] for r in x['rendered_candidates'])
