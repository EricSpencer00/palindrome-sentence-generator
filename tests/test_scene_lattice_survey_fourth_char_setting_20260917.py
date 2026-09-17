import json
from pathlib import Path
from experiments.scene_lattice_survey_fourth_char_setting_20260917 import run
def test_fourth_char_matched_branch():
 x=run(); assert x["novelty_preflight"]["status"]=="passed"; assert x["candidate_count"]==4; assert x["exact_count"]==0
 assert x["stats"]["fourth_character_matches"]==4; assert all(r["fourth_character_obligation"]["matched_fourth_character"] for r in x["rendered_candidates"])
 assert all(not any(r["anti_shortcut_flags"].values()) for r in x["rendered_candidates"]); assert all(r["audit"]["sha256_forward"]!=r["audit"]["sha256_reverse"] for r in x["rendered_candidates"])
