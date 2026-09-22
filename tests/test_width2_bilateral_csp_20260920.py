from experiments.width2_bilateral_csp_20260920 import audit, run


def test_width2_equation_prunes_after_width1():
    result = run()
    assert result["stats"]["width1_survivors"] == 4374
    assert result["stats"]["width2_survivors"] == 1458
    assert result["stats"]["fresh_exact_gt38"] == 0
    assert all(row["character_equation"]["matched"] for row in result["rendered_candidates"])
    assert all(not row["provenance"]["finished_tape_reversal"] for row in result["rendered_candidates"])


def test_width2_lane_full_audit_is_separate():
    assert audit("A man, a plan, a canal: Panama!")["exact"] is True
