from experiments.character_cfg_intersection_live_20260920 import audit, run

def test_live_intersection_emits_audited_prose_controls():
 r=run(); assert r["novelty_preflight"]["status"]=="passed"; assert r["diagnostic_controls"]
 assert all(x["provenance"]["complete_prose"] for x in r["diagnostic_controls"])
 assert all("bilateral_obligation_trace" in x for x in r["diagnostic_controls"])

def test_audit_is_independent():
 x=audit("A man, a plan, a canal: Panama."); assert x["pointer_exact"]; assert x["sha256_forward"]==x["sha256_reverse"]
