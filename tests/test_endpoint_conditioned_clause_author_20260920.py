from endpoint_conditioned_clause_author_20260920 import audit, run


def test_endpoint_condition_is_only_a_pruning_equation():
    result = run()
    assert result["stats"]["endpoint_pairs"] == 2592
    assert result["stats"]["fresh_exact_gt38"] == 0
    assert result["rendered_candidates"]
    for row in result["rendered_candidates"]:
        assert row["endpoint_equation"]["matched"] is True
        assert row["provenance"]["finished_tape_reversal"] is False
        assert row["provenance"]["post_hoc_repair"] is False


def test_endpoint_lane_full_audit_remains_independent():
    assert audit("A man, a plan, a canal: Panama!")["exact"] is True
