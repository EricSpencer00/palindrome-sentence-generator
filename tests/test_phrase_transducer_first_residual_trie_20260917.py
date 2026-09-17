from experiments.phrase_transducer_first_residual_trie_20260917 import run
def test_first_residual_trie():
 x=run();assert x['candidate_count']==8 and x['exact_count']==0
 assert x['novelty_preflight']['status']=='passed';assert all(r['live_obligations']['rendered_after_constraints'] for r in x['rendered_candidates'])
 assert all(not any(r['anti_shortcut_flags'].values()) for r in x['rendered_candidates'])
