from experiments.luna_dependency_scene_csp_20260920 import normalize, solve


def test_all_rendered_candidates_are_exact_under_independent_normalizer():
    result = solve()
    assert result["candidates"] == []
    assert result["closure"] == "no-closure"
    for row in result["attempts"]:
        assert row["exact"] is False
        assert normalize(row["rendered"]) == row["normalized"]
        assert row["normalized"] != row["normalized"][::-1]


def test_dependency_scene_csp_is_not_repair_metadata():
    result = solve()
    assert "dependency" in result["method"]
    assert "repair" not in result["method"]
    assert all(row["frame"]["slots"] for row in result["attempts"])
