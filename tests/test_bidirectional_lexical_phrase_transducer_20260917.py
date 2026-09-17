from experiments.bidirectional_lexical_phrase_transducer_20260917 import run
def test_whole_tape_transducer():
 x=run();assert x['candidate_count']==4 and x['exact_count']==0
 assert x['novelty_preflight']['status']=='passed'
 assert all(not any(r['anti_shortcut_flags'].values()) for r in x['rendered_candidates'])
 assert all(r['live_obligations']['rendered_after_constraints'] for r in x['rendered_candidates'])
