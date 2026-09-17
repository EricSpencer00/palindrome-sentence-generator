import importlib.util
from pathlib import Path
ROOT=Path(__file__).parents[1];spec=importlib.util.spec_from_file_location("s",ROOT/"experiments/scene_lattice_survey_third_char_setting_20260917.py");mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
def test_v_setting_is_gated_to_survey_branch():
 o=mod.run();assert o["novelty_preflight"]["status"]=="passed";assert o["novelty_preflight"]["signature_collision"] is False;assert o["candidate_count"]==4;assert o["exact_count"]==0;assert all(r["third_character_obligation"]["matched_third_character"] for r in o["rendered_candidates"])
 r=o["rendered_candidates"][0];assert r["provenance"]["only_matched_branch"];assert r["provenance"]["syntax_expanded"] is False;assert r["audit"]["sha256_forward"]!=r["audit"]["sha256_reverse"];assert all(v is False for v in r["anti_shortcut_flags"].values())
