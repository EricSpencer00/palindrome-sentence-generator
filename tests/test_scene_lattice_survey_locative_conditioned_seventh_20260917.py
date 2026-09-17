from experiments.scene_lattice_survey_locative_conditioned_seventh_20260917 import run
def test_conditioned_seventh():
 x=run(); assert x['candidate_count']==4 and x['exact_count']==0 and x['stats']['seventh_character_matches']==4
 assert all(r['seventh_character_obligation']['conditioned_locative_noun']=='dells' for r in x['rendered_candidates'])
