from experiments.scene_lattice_survey_locative_morphology_eighth_20260917 import run
def test_morphology_eighth_probe():
 x=run(); assert x['candidate_count']==4 and x['exact_count']==0 and x['stats']['eighth_matches']==4
 assert x['stats']['seventh_matches']==0
