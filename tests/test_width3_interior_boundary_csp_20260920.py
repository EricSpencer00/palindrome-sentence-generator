from width3_interior_boundary_csp_20260920 import audit, run


def test_interior_boundary_equation_prunes_real_slots():
    result = run()
    assert result["stats"]["width1_survivors"] == 12288
    assert result["stats"]["width3_survivors"] == 3072
    assert result["stats"]["interior_boundary_survivors"] == 128
    assert result["stats"]["repeated_content_rows"] == 0
    assert result["stats"]["fresh_exact_gt38"] == 0
    assert all(row["equations"]["interior_verb_boundary"]["matched"] for row in result["rendered_candidates"])
    assert all(row["equations"]["interior_object_boundary"]["matched"] for row in result["rendered_candidates"])


def test_interior_boundary_lane_audits_full_tape():
    assert audit("A man, a plan, a canal: Panama!")["exact"] is True
