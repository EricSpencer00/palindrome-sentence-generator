from experiments.scene_lattice_survey_locative_eighth_20260917 import run
def test_conditioned_eighth():
 x=run(); assert x['candidate_count']==4 and x['exact_count']==0 and x['stats']['eighth_character_matches']==0
 assert all(r['eighth_character_obligation']['conditioned_continuation']=='amid fog' for r in x['rendered_candidates'])
