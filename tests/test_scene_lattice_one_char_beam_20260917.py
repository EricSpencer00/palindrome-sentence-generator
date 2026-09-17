import importlib.util
from pathlib import Path
ROOT=Path(__file__).parents[1];spec=importlib.util.spec_from_file_location("b",ROOT/"experiments/scene_lattice_one_char_beam_20260917.py");mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
def test_one_char_beam_is_bounded_and_audited():
 o=mod.run();assert o["novelty_preflight"]["status"]=="passed";assert o["novelty_preflight"]["signature_collision"] is False;assert o["pre_filter_count"]==96;assert o["candidate_count"]==8;assert o["stats"]["beam_width"]==2;assert o["exact_count"]==0
 r=o["rendered_candidates"][0];assert r["provenance"]["one_char_beam_before_render"];assert r["provenance"]["syntax_expanded"] is False;assert r["audit"]["sha256_forward"]!=r["audit"]["sha256_reverse"];assert all(v is False for v in r["anti_shortcut_flags"].values())
