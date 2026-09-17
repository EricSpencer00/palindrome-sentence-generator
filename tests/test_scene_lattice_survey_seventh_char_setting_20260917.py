from experiments.scene_lattice_survey_seventh_char_setting_20260917 import run
def test_seventh_char_probe():
 x=run(); assert x['candidate_count']==4 and x['exact_count']==0 and x['stats']['seventh_character_matches']==0
 assert all(not any(r['anti_shortcut_flags'].values()) for r in x['rendered_candidates'])
