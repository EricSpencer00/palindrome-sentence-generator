from experiments.phrase_transducer_lexical_aspect_adverb_20260917 import run
def test_lexical_aspect_adverb():
 x=run();assert x['candidate_count']==2 and x['exact_count']==0 and x['stats']['joint_states']==2
 assert all(r['live_obligations']['joint_features_checked_before_render'] for r in x['rendered_candidates'])
