from experiments.agreement_seam_bridge_20260918 import run, audit

def test_agreement_seam_is_new_complete_clause_lane():
    result = run()
    assert result["stats"]["paired_states"] == 20
    assert result["stats"]["exact"] == 0
    assert all(";" in row["rendered"] for row in result["rendered_candidates"])
    assert all(row["provenance"]["complete_intact_clauses"] for row in result["rendered_candidates"])
    assert all(row["novelty_preflight"]["punctuation_carries_letters"] is False for row in result["rendered_candidates"])
    assert all(row["audit"]["sha256_forward"] != row["audit"]["sha256_reverse"] for row in result["rendered_candidates"])

def test_audit_is_independent_two_pointer():
    assert audit("A man, a plan, a canal: Panama!")["two_pointer_exact"]
