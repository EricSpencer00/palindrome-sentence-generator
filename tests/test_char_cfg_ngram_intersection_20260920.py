from experiments.char_cfg_ngram_intersection_20260920 import audit, consume, grammar, run


def test_live_consume_and_cfg_plans():
    assert consume("an", "diana") == ("", "dia")
    assert consume("an", "river") is None
    assert any("rel" in plan[0] for plan in grammar())


def test_bounded_run_keeps_prior_ordering_separate_from_exact_audit():
    result = run(state_limit=20_000, beam=80)
    assert result["novelty_preflight"]["status"] == "passed"
    assert result["novelty_preflight"]["ngram_only_ordering"] is True
    assert result["stats"]["states"] <= 20_000
    assert result["stats"]["pruned"] > 0
    for row in result["complete_prose_controls"]:
        assert audit(row["rendered"]) == row["audit"]
    for row in result["exact_candidates"]:
        assert row["audit"]["exact"] and row["audit"]["letters"] >= 38
