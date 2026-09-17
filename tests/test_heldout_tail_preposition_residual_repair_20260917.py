import importlib.util
from pathlib import Path
ROOT=Path(__file__).parents[1];spec=importlib.util.spec_from_file_location("t",ROOT/"experiments/heldout_tail_preposition_residual_repair_20260917.py");mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
def test_single_tail_repair_has_preflight_and_audit():
 o=mod.run();assert o["novelty_preflight"]["status"]=="passed";assert o["novelty_preflight"]["signature_collision"] is False;assert o["candidate_count"]==2*2*4*2;assert o["exact_count"]==0;assert o["stats"]["longest_letters"]>=100
 r=o["rendered_candidates"][0];assert r["audit"]["sha256_forward"]!=r["audit"]["sha256_reverse"];assert all(v is False for v in r["anti_shortcut_flags"].values())
