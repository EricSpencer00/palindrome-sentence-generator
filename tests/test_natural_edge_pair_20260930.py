from experiments.natural_edge_pair_20260930 import online_pair, run


def test_online_pair_admits_only_a_closed_frontier():
    result = online_pair(
        __import__("experiments.natural_edge_pair_20260930", fromlist=["Clause"]).Clause("Nora, I saw drawer.", "x", "x"),
        __import__("experiments.natural_edge_pair_20260930", fromlist=["Clause"]).Clause("Reward was I, Aron.", "y", "y"),
    )
    assert result["complete"]
    assert result["matched"] == len("noraisawdrawer")


def test_result_has_independent_audit_and_residual():
    result = run()
    candidate = result["candidate"]
    assert candidate["letters"] > 38
    assert candidate["exact_two_pointer"]
    assert candidate["validator"]
    assert candidate["sha256"] == candidate["independent_forward_reverse_sha256"]
    assert result["residual_frontier"]
