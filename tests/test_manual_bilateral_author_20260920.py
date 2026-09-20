from manual_bilateral_author_20260920 import audit, run


def test_manual_bilateral_lane_keeps_both_sides_forward_and_audited():
    result = run()
    assert result["stats"]["rendered_candidates"] == 63504
    assert result["stats"]["fresh_exact_gt38"] == 0
    assert result["rendered_candidates"]
    for row in result["rendered_candidates"]:
        assert row["provenance"]["finished_tape_reversal"] is False
        assert row["provenance"]["post_hoc_repair"] is False
        assert len(row["audit"]["sha256_forward"]) == 64


def test_manual_lane_uses_letter_normalization():
    assert audit("A man, a plan, a canal: Panama!")["exact"] is True
