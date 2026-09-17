from experiments.phrase_transducer_joint_object_agreement_20260917 import run
def test_joint_object_agreement():
 x=run();assert x['candidate_count']==4 and x['exact_count']==0 and x['stats']['joint_states']==2
 assert all(r['live_obligations']['joint_agreement_checked_before_render'] for r in x['rendered_candidates'])
