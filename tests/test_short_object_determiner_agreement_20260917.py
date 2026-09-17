import importlib.util
from pathlib import Path
ROOT=Path(__file__).parents[1];spec=importlib.util.spec_from_file_location("d",ROOT/"experiments/short_object_determiner_agreement_20260917.py");mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
def test_determiner_repair_keeps_short_center_and_audits():
 o=mod.run();assert o["novelty_preflight"]["status"]=="passed";assert o["novelty_preflight"]["signature_collision"] is False;assert o["candidate_count"]==216;assert o["exact_count"]==0;assert o["stats"]["midpoint_inside_token"]==216
 r=o["rendered_candidates"][0];assert r["provenance"]["determiner_realized_before_emission"];assert r["audit"]["sha256_forward"]!=r["audit"]["sha256_reverse"];assert all(v is False for v in r["anti_shortcut_flags"].values())
