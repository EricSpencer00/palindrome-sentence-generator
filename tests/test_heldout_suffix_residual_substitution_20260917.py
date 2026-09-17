import importlib.util
from pathlib import Path
ROOT=Path(__file__).parents[1]; spec=importlib.util.spec_from_file_location("s",ROOT/"experiments/heldout_suffix_residual_substitution_20260917.py"); mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
def test_residual_conditioned_substitutions():
 o=mod.run(); assert o["candidate_count"]==24*3; assert o["exact_count"]==0; assert o["stats"]["longest_letters"]>=100
 r=o["rendered_candidates"][0]; assert r["repair"]["conditioned_on"] is not None; assert r["audit"]["sha256_forward"]!=r["audit"]["sha256_reverse"]
 assert all(v is False for v in r["anti_shortcut_flags"].values())
def test_surface_alternatives_are_not_spelling_edits():
 assert mod.ALTS["present_sg"]==("examines","inspects","reviews")
