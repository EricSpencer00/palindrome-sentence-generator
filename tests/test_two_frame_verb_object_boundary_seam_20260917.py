import importlib.util
from pathlib import Path
ROOT=Path(__file__).parents[1];spec=importlib.util.spec_from_file_location("v",ROOT/"experiments/two_frame_verb_object_boundary_seam_20260917.py");mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
def test_verb_object_boundary_rejects_rewrites_and_audits():
 o=mod.run();assert o["novelty_preflight"]["status"]=="passed";assert o["novelty_preflight"]["signature_collision"] is False;assert o["candidate_count"]==4;assert o["exact_count"]==0;assert o["stats"]["midpoint_inside_token"]==4
 r=o["rendered_candidates"][0];assert r["provenance"]["malformed_reordering_rejected"];assert r["audit"]["sha256_forward"]!=r["audit"]["sha256_reverse"];assert all(v is False for v in r["anti_shortcut_flags"].values())
