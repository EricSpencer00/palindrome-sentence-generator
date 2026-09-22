from experiments.clause_boundary_character_chart_20260921 import run


def test_clause_boundary_chart_contract():
    data = run()
    assert data["stats"]["heldout_skeletons"] == 4
    assert data["stats"]["chart_pairs"] == 16
    assert data["provenance"]["heldout_grammar"]
    assert data["provenance"]["bounded_deterministic_search"]
    assert data["novelty_preflight"]["status"] == "passed"
    assert data["construction_queue"][0]["status"] == "complete"
    assert all(row["provenance"]["outside_in_head_emission"] for row in data["controls"])
    assert all(row["provenance"]["finished_tape_reversal"] is False for row in data["controls"])
    assert all("audit" in row and "chart" in row for row in data["controls"])
    assert all("while watch" not in row["rendered"] for row in data["controls"])
