from experiments.multiclause_live_seam_chart_20260917 import run


def test_multiclause_chart_preserves_distinct_diagnostics_and_zero_exact():
    artifact = run()
    assert artifact["stats"]["typed_clauses"] == 194
    assert artifact["stats"]["compositions"] == 1_500_660
    assert artifact["stats"]["exact"] == 0
    assert artifact["stats"]["nonrepeating_diagnostics"] == 20
    assert all(not row["audit"]["exact"] for row in artifact["candidates"])
    assert all(not row["anti_shortcut"]["repeated_clause"] for row in artifact["candidates"])
