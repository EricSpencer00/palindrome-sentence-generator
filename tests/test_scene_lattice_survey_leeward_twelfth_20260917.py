from experiments.scene_lattice_survey_leeward_twelfth_20260917 import run
def test_leeward_twelfth():
 x=run();assert x['candidate_count']==4 and x['exact_count']==0 and x['stats']['twelfth_matches']==4
 assert all(r['twelfth_character_obligation']['conditioned_continuation']=='leeward' for r in x['rendered_candidates'])
