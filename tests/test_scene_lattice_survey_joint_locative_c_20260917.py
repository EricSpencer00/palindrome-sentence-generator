from experiments.scene_lattice_survey_joint_locative_c_20260917 import run
def test_c_tail():
 x=run();assert x['candidate_count']==4 and x['exact_count']==0 and x['stats']['ninth_matches']==4
 assert all(r['ninth_character_obligation']['conditioned_tail']=='Lacy' for r in x['rendered_candidates'])
