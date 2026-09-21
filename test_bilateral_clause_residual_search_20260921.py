from bilateral_clause_residual_search_20260921 import audit, residual, run

def test_pointer_sha_audit_and_residual_solver():
    x = audit("A man, a plan, a canal: Panama!")
    assert x["pointer_exact"] and x["sha256_forward"] == x["sha256_reverse"]
    assert residual("abc", "cba")["closed"]

def test_candidates_have_complete_units_and_no_shortcuts():
    result = run()
    assert result["stats"]["controls"] == 16
    assert result["novelty_preflight"]["status"] == "passed"
    for row in result["rendered_candidates"]:
        assert row["provenance"]["residual_solved_before_rendering"]
        assert row["gates"]["complete_typed_clauses"]
        assert row["gates"]["no_lm_reward"]
