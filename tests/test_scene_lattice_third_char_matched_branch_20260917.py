import importlib.util
from pathlib import Path
ROOT=Path(__file__).parents[1];spec=importlib.util.spec_from_file_location("t",ROOT/"experiments/scene_lattice_third_char_matched_branch_20260917.py");mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
def test_third_char_is_gated_to_two_char_branches():
 o=mod.run();assert o["novelty_preflight"]["status"]=="passed";assert o["novelty_preflight"]["signature_collision"] is False;assert o["two_char_input_count"]>0;assert o["candidate_count"]==o["two_char_input_count"];assert o["exact_count"]==0
 r=o["rendered_candidates"][0];assert r["provenance"]["third_char_checked_only_on_two_char_matches"];assert r["provenance"]["syntax_expanded"] is False;assert r["audit"]["sha256_forward"]!=r["audit"]["sha256_reverse"];assert all(v is False for v in r["anti_shortcut_flags"].values())
