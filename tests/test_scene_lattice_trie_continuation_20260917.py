import importlib.util
from pathlib import Path
ROOT=Path(__file__).parents[1];spec=importlib.util.spec_from_file_location("t",ROOT/"experiments/scene_lattice_trie_continuation_20260917.py");mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
def test_trie_filter_is_live_and_independent():
 o=mod.run();assert o["novelty_preflight"]["status"]=="passed";assert o["novelty_preflight"]["signature_collision"] is False;assert o["pre_filter_count"]==96;assert 0<o["candidate_count"]<=o["pre_filter_count"];assert o["exact_count"]==0
 r=o["rendered_candidates"][0];assert r["continuation"]["trie_continuation"];assert r["provenance"]["filter_before_final_render"];assert r["provenance"]["syntax_expanded"] is False;assert r["audit"]["sha256_forward"]!=r["audit"]["sha256_reverse"];assert all(v is False for v in r["anti_shortcut_flags"].values())
