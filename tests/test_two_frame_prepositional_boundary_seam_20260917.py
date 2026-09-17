import importlib.util
from pathlib import Path
ROOT=Path(__file__).parents[1];spec=importlib.util.spec_from_file_location("p",ROOT/"experiments/two_frame_prepositional_boundary_seam_20260917.py");mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
def test_preposition_boundary_is_ordinary_and_audited():
 o=mod.run();assert o["novelty_preflight"]["status"]=="passed";assert o["novelty_preflight"]["signature_collision"] is False;assert o["candidate_count"]==4;assert o["exact_count"]==0;assert o["stats"]["midpoint_inside_token"]==4
 r=o["rendered_candidates"][0];assert r["provenance"]["ordinary_grammar"];assert r["provenance"]["new_lexical_bank"] is False;assert r["audit"]["sha256_forward"]!=r["audit"]["sha256_reverse"];assert all(v is False for v in r["anti_shortcut_flags"].values())
