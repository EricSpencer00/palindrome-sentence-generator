import importlib.util
from pathlib import Path
ROOT=Path(__file__).parents[1];spec=importlib.util.spec_from_file_location("a",ROOT/"experiments/short_second_object_adjective_20260917.py");mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
def test_second_adjective_only_keeps_short_center_and_audits():
 o=mod.run();assert o["novelty_preflight"]["status"]=="passed";assert o["novelty_preflight"]["signature_collision"] is False;assert o["candidate_count"]==23328;assert o["exact_count"]==0;assert o["stats"]["midpoint_inside_token"]==23328
 r=o["rendered_candidates"][0];assert r["provenance"]["second_adjective_realized_before_emission"];assert r["audit"]["sha256_forward"]!=r["audit"]["sha256_reverse"];assert all(v is False for v in r["anti_shortcut_flags"].values())
