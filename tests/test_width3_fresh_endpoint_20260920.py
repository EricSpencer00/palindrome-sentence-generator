from width3_fresh_endpoint_20260920 import audit, run


def test_width3_equation_is_incremental_and_prunes():
    result = run()
    assert result["stats"]["width1_survivors"] == 729
    assert result["stats"]["width2_survivors"] == 729
    assert result["stats"]["width3_survivors"] == 243
    assert result["stats"]["fresh_exact_gt38"] == 0
    assert all(row["endpoint_equation"]["matched"] for row in result["rendered_candidates"])


def test_width3_lane_audits_full_rendered_tape():
    assert audit("A man, a plan, a canal: Panama!")["exact"] is True
