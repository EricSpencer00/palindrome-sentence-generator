from experiments.three_clause_center_crossing_20260917 import run


def test_three_clause_chart_excludes_repeated_controls():
    artifact = run()
    assert artifact["stats"]["typed_clauses"] == 194
    assert artifact["stats"]["repeated_controls_rejected"] == 49_430
    assert artifact["stats"]["exact"] == 0
    assert artifact["stats"]["center_crossings"] == 0
    assert all(not row["repeated_clause"] for row in artifact["candidates"])
    assert all(not row["audit"]["exact"] for row in artifact["candidates"])
