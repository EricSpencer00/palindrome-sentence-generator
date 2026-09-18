from experiments.agreement_seam_bridge_20260918 import audit, run

def test_independent_audit_and_provenance():
    p = run()
    assert p["construction"]["live_character_constraints_before_render"]
    assert p["construction"]["independent_audit"] == "two_pointer_and_sha256"
    assert p["novelty_preflight"]["prior_lane_reused"] is False
    assert all("rendered" in x and "audit" in x for x in p["rendered_candidates"])
    assert all(x["provenance"]["catalogue_used"] is False for x in p["rendered_candidates"])

def test_audit_rejects_non_palindrome():
    assert audit("A baker marks maps") ["two_pointer_exact"] is False
    assert audit("Live on; no evil.")["two_pointer_exact"] is True
