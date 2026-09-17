from experiments.phrase_transducer_joint_tense_aspect_20260917 import run
def test_joint_tense_aspect():
 x=run();assert x['candidate_count']==6 and x['exact_count']==0 and x['stats']['joint_feature_states']==3
 assert all(r['live_obligations']['joint_features_checked_before_render'] for r in x['rendered_candidates'])
