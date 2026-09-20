from experiments.bottom_up_cfg_character_intersection_20260920 import audit, chart, consume, run


def test_chart_is_bottom_up_and_residual_is_character_level():
    c = chart()
    assert all(c[k] for k in ("NP", "VP", "PP", "REL"))
    assert consume("an", "diana") == ("", "dia")
    assert consume("an", "river") is None


def test_complete_controls_and_independent_audit():
    result = run(state_limit=20000)
    assert result["novelty_preflight"]["status"] == "passed"
    assert result["stats"]["combines"] > 0
    assert result["complete_clause_derivations"] > 0
    for row in result["complete_prose_controls"]:
        assert audit(row["rendered"]) == row["audit"]
