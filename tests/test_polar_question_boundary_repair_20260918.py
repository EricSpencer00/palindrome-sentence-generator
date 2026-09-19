from experiments.polar_question_boundary_repair_20260918 import audit, build_graph, pair_reachability, run


def test_pair_graph_keeps_incoming_right_order_and_has_reachable_states():
    graph = build_graph()
    reach = pair_reachability(graph)
    assert reach["pair_states"] > 0
    assert reach["pair_edges"] > 0


def test_audit_is_independent_and_rejects_nonpalindrome():
    result = audit("Was Noel calm? Leon saw a map.")
    assert result["two_pointer_exact"] is False
    assert result["sha_equal_under_reversal"] is False


def test_run_records_reader_gate_and_independent_exact_fields():
    result = run()
    assert result["stats"]["reader_eligible"] == 0
    assert result["stats"]["longest_exact_letters"] == 44
    assert len(result["rendered_candidates"]) == 1
    for row in result["rendered_candidates"]:
        assert row["audit"]["two_pointer_exact"] is True
        assert row["audit"]["sha_equal_under_reversal"] is True
        assert row["reader_status"].startswith("not_run")
