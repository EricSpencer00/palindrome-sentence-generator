from experiments.scene_lattice_survey_high_thirteenth_20260917 import run
def test_high_thirteenth():
 x=run();assert x['candidate_count']==4 and x['exact_count']==0 and x['stats']['thirteenth_matches']==0
