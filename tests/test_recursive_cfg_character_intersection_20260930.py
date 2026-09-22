from experiments.recursive_cfg_character_intersection_20260930 import audit, run


def test_recursive_cfg_search_records_online_frontier_and_novelty_gate():
    result = run(max_states=20_000)
    assert result["novelty_preflight"]["status"] == "passed"
    assert result["stats"]["states"] > 0
    assert result["provenance"]["online_pruning"]
    assert result["residual_frontier"]


def test_independent_audit_has_two_pointer_and_forward_reverse_sha():
    result = audit("A man, a plan, a canal: Panama")
    assert result["pointer_exact"]
    assert result["sha256_forward"] == result["sha256_reverse"]
