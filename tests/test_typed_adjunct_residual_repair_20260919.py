from experiments.typed_adjunct_residual_repair_20260919 import run


def test_adjunct_repair_records_a_new_constructive_lane_without_promoting_seed():
    result = run()
    assert result["method"].startswith("agreement-safe adjunct")
    assert result["stats"]["longest_letters"] > 38
    assert result["stats"]["mechanically_admitted"] == 0
    assert result["stats"]["longest_exact_letters"] == 38
    assert not any(row["adjunct_edge"] for row in result["exact_candidates"])
    assert any(row.get("control") and row["audit"]["letters"] > 38 for row in result["rows"])
