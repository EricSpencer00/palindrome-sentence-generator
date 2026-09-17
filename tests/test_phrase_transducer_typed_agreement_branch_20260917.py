from experiments.phrase_transducer_typed_agreement_branch_20260917 import run
def test_typed_agreement_branch():
 x=run();assert x['candidate_count']==8 and x['exact_count']==0 and x['stats']['agreement_states']==2
 assert all(r['live_obligations']['agreement_checked_before_render'] for r in x['rendered_candidates'])
 assert all(not any(r['anti_shortcut_flags'].values()) for r in x['rendered_candidates'])
