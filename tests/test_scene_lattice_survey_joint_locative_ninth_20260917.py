from experiments.scene_lattice_survey_joint_locative_ninth_20260917 import run
def test_ninth_tail_probe():
 x=run();assert x['candidate_count']==4 and x['exact_count']==0 and x['stats']['ninth_matches']==0
