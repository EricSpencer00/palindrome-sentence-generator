from experiments.variable_bilateral_clause_decoder_20260920 import audit, paths, run


def test_variable_paths_include_held_out_clause_roles():
    ps = paths()
    assert len({len(p) for p in ps}) >= 3
    assert any("REL" in p for p in ps)
    assert any("PP" in p for p in ps)


def test_run_has_complete_clause_provenance_and_independent_audit():
    result = run(state_limit=20_000)
    assert result["novelty_preflight"]["status"] == "passed"
    assert result["stats"]["transitions"] > 0
    for row in result["exact_candidates"]:
        assert row["provenance"]["complete_semantic_clauses"]
        assert audit(row["rendered"]) == row["audit"]
