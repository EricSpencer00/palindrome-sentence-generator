from experiments.synchronized_interior_csp_20260920 import a, run


def test_synchronized_csp_does_not_overclaim_character_matching():
    result = run()
    assert result["stats"]["endpoint_survivors"] == 8192
    assert result["stats"]["interior_csp_survivors"] == 8192
    assert result["stats"]["repeated_content_rows"] == 3872
    assert result["stats"]["fresh_exact_gt38"] == 0
    for row in result["rendered_candidates"]:
        assert row["csp"]["interior_width"] == 0
        assert row["csp"]["interior_constraint"] == "four aligned grammatical roles"
        assert row["provenance"]["finished_tape_reversal"] is False


def test_synchronized_lane_uses_independent_full_audit():
    assert a("A man, a plan, a canal: Panama!")["exact"] is True
