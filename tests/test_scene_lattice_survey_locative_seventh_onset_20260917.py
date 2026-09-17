from experiments.scene_lattice_survey_locative_seventh_onset_20260917 import run
def test_locative_seventh_onset():
 x=run(); assert x['candidate_count']==4 and x['exact_count']==0
 assert all(r['choices']['setting_role']=='locative' for r in x['rendered_candidates'])
 assert all(not any(r['anti_shortcut_flags'].values()) for r in x['rendered_candidates'])
