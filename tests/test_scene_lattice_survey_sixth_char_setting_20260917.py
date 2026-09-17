from experiments.scene_lattice_survey_sixth_char_setting_20260917 import run
def test_sixth_char_matched_branch():
 x=run(); assert x['candidate_count']==4 and x['exact_count']==0 and x['stats']['sixth_character_matches']==4
 assert all(r['sixth_character_obligation']['matched_sixth_character'] for r in x['rendered_candidates'])
 assert all(not any(r['anti_shortcut_flags'].values()) for r in x['rendered_candidates'])
