from experiments.outer_domain_compatibility_20260921 import run, outer_support


def test_outer_operator_is_bounded_and_audited():
    s = outer_support()
    assert s["pair_count"] == 36
    assert s["conflict_pairs"] > 0
    result = run(limit=5000)
    assert result["provenance"]["target_length_mirror_propagation"]
    assert result["provenance"]["dependency_grammar_preserved"]
    assert result["novelty_preflight"]["status"] == "passed"
    assert result["novelty_preflight"]["catalogue_used"] is False
    assert result["counts"]["exact_outputs"] == 0
