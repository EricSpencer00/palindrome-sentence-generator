from experiments.inflectional_boundary_graph_20260921 import audit, run

def test_inflectional_graph_is_bounded_and_independently_audited():
    result = run()
    assert result["stats"]["frames"] == 4
    assert result["novelty_preflight"]["status"] == "passed"
    for row in result["rendered_outputs"]:
        assert row["audit"] == audit(row["rendered"])
        assert row["provenance"]["inflection_carried_in_state"]
        assert row["provenance"]["complete_clause"]
        assert not row["provenance"]["posthoc_repair"]
        assert len(row["audit"]["sha256_forward"]) == 64
