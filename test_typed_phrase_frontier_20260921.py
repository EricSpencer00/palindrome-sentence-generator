from typed_phrase_frontier_20260921 import audit, run

def test_frontier_emits_controls_and_provenance():
    result = run()
    assert result["novelty_preflight"]["status"] == "passed"
    assert result["rendered_controls"]
    assert all(row["provenance"]["online_debt_discharge"] for row in result["rendered_controls"])

def test_independent_audit_and_exact_closure():
    result = run()
    assert audit("Able was I ere I saw Elba.")["pointer_exact"]
    assert result["exact_candidates"]
    assert all(row["audit"]["sha256_forward"] == row["audit"]["sha256_reverse"] for row in result["exact_candidates"])
