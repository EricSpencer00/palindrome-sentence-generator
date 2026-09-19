from experiments.polar_question_token_boundary_csp_20260918 import audit, build_graph, run


def test_graph_has_independent_answer_token_paths():
    graph = build_graph()
    assert graph.size > 100
    assert any("fronted subject" in edge[4] for edges in graph.out.values() for edge in edges)


def test_audit_has_independent_hashes():
    row = audit("Was Noel calm? A note, Leon saw.")
    assert row["two_pointer_exact"] is False
    assert row["sha_equal_under_reversal"] is False


def test_run_has_explicit_reader_gate():
    result = run()
    assert result["stats"]["reader_eligible"] == 0
    assert result["non_exact_reconstruction_paths"]
    for row in result["rendered_candidates"]:
        assert row["audit"]["two_pointer_exact"] is True
        assert row["audit"]["sha_equal_under_reversal"] is True
