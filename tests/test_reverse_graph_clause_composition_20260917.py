from experiments.reverse_graph_clause_composition_20260917 import run

def test_reverse_graph_is_live_and_fail_closed():
    out = run()
    assert out["search_integrity"]["live_character_pruning"]
    assert out["search_integrity"]["posthoc_reversal"] is False
    assert out["stats"]["graph_products"] == 23328
    for row in out["rows"]:
        assert row["provenance"]["agreement_checked"]
        assert "rendered" in row and "audit" in row

def test_independent_audit_fields():
    for row in run()["rows"]:
        a=row["audit"]
        assert a["two_pointer_exact"] == a["exact"]
        assert len(a["sha256"]) == 64 and len(a["reverse_sha256"]) == 64
