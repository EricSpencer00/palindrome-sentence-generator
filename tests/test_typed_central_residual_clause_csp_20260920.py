from experiments.typed_central_residual_clause_csp_20260920 import audit, run

def test_typed_center_search_is_bounded_and_retains_controls():
    result = run()
    assert result["stats"]["assignments"] == 18
    assert result["stats"]["residual_states"] > 0
    assert result["stats"]["complete_controls"] == 2
    assert result["stats"]["fresh_exact_gt38"] == 0
    assert all(x["provenance"]["complete_typed_center"] for x in result["rendered_candidates"])

def test_seed_audit_remains_independent():
    a = audit("An aide rips nine memos; some men inspire Diana.")
    assert a["exact"] and a["independent_two_pointer"] and a["sha_equal_under_reverse"]
