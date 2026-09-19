from experiments.synchronous_grammar_product_20260919 import audit, inventory, run

def test_typed_inventory_and_preflight():
    result = run(500_000)
    assert result["novelty_preflight"]["status"] == "passed"
    assert result["config"]["inventory"] == len(inventory())
    assert set(result["config"]["families"]) == {"transitive", "copular", "locative"}
    assert result["stats"]["truncated"] is False

def test_independent_exactness_audit():
    result = run(500_000)
    assert result["provenance"]["independent_audits"]
    assert result["stats"]["exact_rows"] == len(result["exact_candidates"])
    assert audit("level")["two_pointer_exact"]
    assert audit("ordinary")["two_pointer_exact"] is False

def test_bounded_search_is_honest_about_frontier():
    result = run(2)
    assert result["stats"]["truncated"] is True
    assert "no-closure hypothesis" in result["falsifier"]
