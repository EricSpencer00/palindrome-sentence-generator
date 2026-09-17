from experiments.scene_lattice_survey_joint_locative_tenth_20260917 import run
def test_tenth_tail():
 x=run();assert x['candidate_count']==4 and x['exact_count']==0 and x['stats']['tenth_matches']==4
