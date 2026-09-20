from experiments.semantic_role_bilateral_20260920 import run


def test_semantic_role_lane_is_bounded_and_provenanced():
    result = run(max_nodes=200)
    assert result["stats"]["status"] in {"SAT", "UNSAT", "timeout"}
    assert result["provenance"]["role_constraints"]
    assert all(row["audit"]["exact"] for row in result["paths"])
