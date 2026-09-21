from experiments.prose_first_scene_voice_lattice_20260921 import audit,run
def test_scene_lattice_has_independent_voice_controls():
 r=run(); assert r["novelty_preflight"]["status"]=="passed"; assert r["rendered_controls"]; assert r["stats"]["live_pruned"]
 assert all(x["provenance"]["shared_semantic_event_graph"] for x in r["rendered_controls"])
def test_audit_hashes():
 a=audit("A man, a plan, a canal, panama."); assert a["pointer_exact"] and a["sha256_forward"]==a["sha256_reverse"]
