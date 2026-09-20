from experiments.chart_phrase_path_20260920 import audit, search

def test_chart_path_audits_and_regression():
    assert audit("A man, a plan, a canal: Panama!")["exact"]
    data = search(limit=4)
    assert data["run_id"] == "chart-phrase-path-20260920"
    assert data["novelty_preflight"]["status"] == "passed"
    assert data["rendered_candidates"] and data["controls"]
    assert all("boundary_state" in row and "audit" in row for row in data["rendered_candidates"])
    assert data["stats"]["chart_paths"] > 0
