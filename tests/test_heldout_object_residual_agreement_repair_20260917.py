import importlib.util
from pathlib import Path
ROOT=Path(__file__).parents[1];spec=importlib.util.spec_from_file_location("o",ROOT/"experiments/heldout_object_residual_agreement_repair_20260917.py");mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
def test_object_repair_retains_agreement_and_audit():
 o=mod.run();assert o["candidate_count"]==2*2*2*5*3*2;assert o["exact_count"]==0;assert o["stats"]["longest_letters"]>=100;assert o["stats"]["residual_conditioned"]==o["candidate_count"]
 r=o["rendered_candidates"][0];assert r["audit"]["sha256_forward"]!=r["audit"]["sha256_reverse"];assert all(v is False for v in r["anti_shortcut_flags"].values());assert r["provenance"]["agreement_state_retained"]
