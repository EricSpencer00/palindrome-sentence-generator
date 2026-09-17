from experiments.scene_lattice_survey_joint_locative_la_20260917 import run
def test_joint_la():
 x=run(); assert x['candidate_count']==4 and x['exact_count']==0 and x['stats']['joint_matches']==4
 assert all(not any(r['anti_shortcut_flags'].values()) for r in x['rendered_candidates'])
