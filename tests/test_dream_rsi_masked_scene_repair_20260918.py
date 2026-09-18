from experiments.dream_rsi_masked_scene_repair_20260918 import audit, repair_pair, run

def test_masked_scene_repair_records_live_equations_and_audit():
    result = run()
    assert result["stats"]["fresh_nodes"] >= 0
    assert result["stats"]["fresh_exact"] == len(result["fresh_exact_closures"])
    assert all(row["provenance"]["fresh_authored_scene"] for row in result["rendered_candidates"])
    assert all("audit" in row and "rendered" in row for row in result["rendered_candidates"])
    assert result["next_repair"]["operator"]

def test_independent_audit_is_exactness_only():
    assert audit("Step on, no pets.")["two_pointer_exact"]
    assert not audit("the baker maps fresh notes near dawn.")["two_pointer_exact"]
    row = repair_pair(("the baker", "maps", "fresh notes", "near dawn"), ("a quiet nurse", "folds", "old maps", "after class"))
    assert row["trace"] and all("equations_checked" in t for t in row["trace"])
