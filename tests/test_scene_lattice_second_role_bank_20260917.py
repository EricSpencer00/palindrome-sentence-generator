import importlib.util
from pathlib import Path
ROOT=Path(__file__).parents[1];spec=importlib.util.spec_from_file_location("b",ROOT/"experiments/scene_lattice_second_role_bank_20260917.py");mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
def test_two_independent_role_banks_filter_frontier():
 o=mod.run();assert o["novelty_preflight"]["status"]=="passed";assert o["novelty_preflight"]["signature_collision"] is False;assert o["pre_filter_count"]==3*4*4*2;assert 0<o["candidate_count"]<=o["pre_filter_count"];assert o["exact_count"]==0
 r=o["rendered_candidates"][0];assert r["provenance"]["object_setting_banks_independent"];assert r["frontier_obligation"]["score"]==o["stats"]["max_frontier_score"];assert r["audit"]["sha256_forward"]!=r["audit"]["sha256_reverse"];assert all(v is False for v in r["anti_shortcut_flags"].values())
