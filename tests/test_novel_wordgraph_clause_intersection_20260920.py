from experiments.novel_wordgraph_clause_intersection_20260920 import audit, intersect, run


def test_wordgraph_run_uses_full_rendered_tape_for_admission():
    result = run()
    assert result["novelty_preflight"]["status"] == "passed"
    assert result["stats"]["fresh_exact_gt38"] == 0
    assert result["rendered_candidates"]
    for row in result["rendered_candidates"]:
        assert row["audit"]["exact"] is False
        assert row["provenance"]["finished_tape_reversal"] is False
        assert row["provenance"]["post_hoc_repair"] is False


def test_wordgraph_intersection_is_an_explicit_seam_diagnostic():
    seam = intersect("abc", "cba")
    assert seam["exact"] is True
    assert audit("abc, and cba.")["exact"] is False
