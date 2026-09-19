from experiments.typed_constituent_seam_search_20260919 import _zipper, run


def test_typed_constituent_lane_recovers_only_the_current_exact_frontier():
    result = run()
    assert result["stats"]["exact"] == 2
    assert result["stats"]["longest_exact_letters"] == 38
    exact = {row["rendered"] for row in result["exact_candidates"]}
    assert "An aide rips nine memos; some men inspire Diana." in exact
    assert "Some men inspire diana; an aide rips nine memos." in exact
    assert all(row["mechanically_admitted"] for row in result["exact_candidates"])


def test_online_zipper_reports_first_cross_boundary_mismatch():
    result = _zipper("the captain reads a note", "some men inspire Diana")
    assert result["closed"] is False
    assert result["first_failure"]["index"] == 0
    assert result["first_failure"]["matched"] is False
